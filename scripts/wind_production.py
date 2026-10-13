import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import config
from scipy.spatial import cKDTree
import functions
from dataclasses import dataclass
from typing import Tuple

def _read_tracker_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, sep=";", decimal=",", encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="latin1",
        encoding_errors="replace",
        low_memory=False,
    )
    
@dataclass(frozen=True)
class MonthSpec:
    year: str
    month_number: str   # "01"
    month_name: str     # "jan"

    @property
    def prod_year(self) -> int:
        return int(self.year)

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
    
    class GridIndexer:
        def __init__(self, lat2d: np.ndarray, lon2d: np.ndarray):
            self.shape = lat2d.shape
            self.n_points = lat2d.size
            
            lon_norm = lon2d.ravel() % 360.0
            lat_flat = lat2d.ravel()
            
            p_main = np.column_stack((lat_flat, lon_norm))
            p_left = np.column_stack((lat_flat, lon_norm - 360.0))
            p_right = np.column_stack((lat_flat, lon_norm + 360.0))
            
            self.tree = cKDTree(np.vstack((p_main, p_left, p_right)))

        def map_points(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            lon_q = lon % 360.0
            pts = np.column_stack((lat, lon_q))
            
            _, idx = self.tree.query(pts, k=1)
            
            idx_orig = idx % self.n_points
            y, x = np.unravel_index(idx_orig, self.shape)
            return y.astype(int), x.astype(int)

    def country_timeseries(
        self,
        ms: MonthSpec,
        xrds: xr.Dataset,
        indexer: GridIndexer,
        ZONES: list[str],
        LOCATION_COLS_WIND: list[str],
        country: str,
        area_code: str,
        return_farm_data: bool = True,

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
                zone_locations = {location.casefold() for location in zone_cities}
                mask = pd.Series(False, index=df_country.index)
                cols_to_check = LOCATION_COLS_WIND
                for col in cols_to_check:
                    if col in df_country.columns:
                        mask |= df_country[col].astype("string").str.casefold().isin(zone_locations)
                df_country = df_country[mask].copy()

        if len(df_country) == 0:
             return (*empty_ret, pd.DataFrame(), []) if return_farm_data else empty_ret

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

        mw_2026 = np.zeros(T, dtype=np.float64)
        mw_asbuilt = np.zeros(T, dtype=np.float64)
        farm_results = []

        for i in tqdm(range(len(df_country)), desc=f"Estimating wind power for {country}", unit="farm"):
            row = df_country.iloc[i]
            ts_mw = functions.estimate_wind_power(
                lat=float(row["Latitude"]),
                lon=float(row["Longitude"]),
                capacity=row["Capacity (MW)"],
                startyear=int(start_year_model[i]),
                installation_type=row["Installation Type"],
                xrds=xrds,
                y_idx=int(y_idx[i]),
                x_idx=int(x_idx[i]),
                wts_smoothing=False,
                spatial_interpolation=True,
                wake_loss_factor=1.0,
                single_turb_curve=False,
                enforce_start_year=False,
                verbose=False,
            )
            if ts_mw is None:
                continue

            mw_2026 += ts_mw
            if asbuilt_mask[i]:
                mw_asbuilt += ts_mw
            
            if return_farm_data:
                farm_results.append((i, ts_mw))

        shifted_index_wind = pd.to_datetime(xrds["valid_time"].values) - pd.Timedelta(hours=1)

        res = (
            mw_asbuilt.astype(np.float64),
            mw_2026.astype(np.float64),
        )
        
        if return_farm_data:
             return (*res, df_country, farm_results)
        return res
