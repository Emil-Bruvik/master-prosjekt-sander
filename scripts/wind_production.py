class WindCalculator:
    def __init__(self):
        df_main = _read_tracker_csv(config.WIND_TRACKER_CSV)
        df_below = _read_tracker_csv(config.WIND_TRACKER_BELOW_CSV)
        self.df = pd.concat([df_main, df_below], ignore_index=True)
        self.ref_startyear = int(pd.to_numeric(self.df["Start year"], errors="coerce").median())

    def open_weather(self, ms: MonthSpec) -> xr.Dataset:
        fn = config.CERRA_MULTI_LEVEL_DIR / ms.year / f"cerra_{ms.year}_multi_level_{ms.month_name}.nc"
        ds = xr.open_dataset(fn, engine="h5netcdf")
        ds.load()
        return ds

    def country_timeseries(
        self,
        ms: MonthSpec,
        xrds: xr.Dataset,
        indexer: GridIndexer,
        actual_loader: ActualGenerationLoader,
        actual_file: pd.DataFrame,
        country: str,
        area_code: str,
        return_farm_data: bool = True,
        fallback_factor: float | None = None,
        force_factor: bool = False,
    ):
        T = xrds.sizes["valid_time"]
        empty_ret = (np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), 0.0)

        df_country = self.df[self.df["Country/Area"] == country].copy()
        if len(df_country) == 0:
             return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        if area_code in ZONES:
            if area_code.startswith('DK'):
                lon_vals = pd.to_numeric(df_country['Longitude'], errors='coerce')
                if area_code == 'DK1':
                    df_country = df_country[lon_vals < 10.9].copy()
                else: #DK2
                    df_country = df_country[lon_vals >= 10.9].copy()
            else:
                zone_cities = functions.get_bidding_zone_mapping(area_code)
                mask = pd.Series(False, index=df_country.index)
                cols_to_check = LOCATION_COLS_PV if "PV" in str(type(self)) else LOCATION_COLS_WIND
                for col in cols_to_check:
                    if col in df_country.columns:
                        mask |= df_country[col].isin(zone_cities)
                df_country = df_country[mask].copy()

        if len(df_country) == 0:
             return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

        wind_actual = actual_loader.wind_series_mw(actual_file, area_code)
        use_fallback = force_factor or wind_actual.sum() <= 1.0

        if use_fallback and fallback_factor is None:
            print(f"No actual production data for {country} (Wind). Using correction factor = 1.0.", flush=True)
            fallback_factor = 1.0

        lat = df_country["Latitude"].to_numpy()
        lon = df_country["Longitude"].to_numpy()
        y_idx, x_idx = indexer.map_points(lat, lon)
        
        df_country['y_idx'] = y_idx
        df_country['x_idx'] = x_idx

        start_year_f = pd.to_numeric(df_country["Start year"], errors="coerce").to_numpy(dtype=float)
        asbuilt_mask = (
            np.isfinite(start_year_f)
            & (start_year_f >= 1800)
            & (start_year_f <= ms.prod_year)
        )
        
        ref = float(self.ref_startyear) if getattr(self, "ref_startyear", None) is not None else float(ms.prod_year)
        start_year_model = np.where(np.isfinite(start_year_f), np.floor(start_year_f), ref).astype(int)

        mw_2025 = np.zeros(T, dtype=np.float64)
        mw_asbuilt = np.zeros(T, dtype=np.float64)
        farm_results = []

        for i in tqdm(range(len(df_country)), desc=f"Estimating wind power for {country}", unit="farm"):
            row = df_country.iloc[i]
            ts_mw = functions.estimate_wind_power(
                country=country,
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                capacity=row["Capacity (MW)"],
                startyear=int(start_year_model[i]),
                prod_year=ms.prod_year,
                status=row["Status"],
                installation_type=row["Installation Type"],
                xrds=xrds,
                y_idx=int(y_idx[i]),
                x_idx=int(x_idx[i]),
                wts_smoothing=False,
                power_smoothing=False,
                wake_loss_factor=1.0,
                spatial_interpolation=True,
                verbose=False,
                single_turb_curve=False,
                enforce_start_year=False,
            )
            if ts_mw is None:
                continue

            mw_2025 += ts_mw
            if asbuilt_mask[i]:
                mw_asbuilt += ts_mw
            
            if return_farm_data:
                farm_results.append((i, ts_mw))

        shifted_index_wind = pd.to_datetime(xrds["valid_time"].values) - pd.Timedelta(hours=1)
        
        if use_fallback:
            factor = float(fallback_factor)
        else:
            factor = PVCalculator._factor_from_asbuilt(
                mw_asbuilt,
                shifted_index_wind,
                wind_actual,
            )

        res = (
            mw_asbuilt.astype(np.float64),
            mw_2025.astype(np.float64),
            float(factor)
        )
        
        if return_farm_data:
             return (*res, df_country, farm_results)
        return res
