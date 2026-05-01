import pandas as pd
import numpy as np
from pathlib import Path

# Path to input folder
path = Path("input")

# Load available files (graceful if missing)
def load_csv(name):
    p = path / name
    return pd.read_csv(p) if p.exists() else None

web = load_csv("web_traffic.csv")
orders = load_csv("orders.csv")
order_items = load_csv("order_items.csv")
payments = load_csv("payments.csv")
geography = load_csv("geography.csv")

# Helpers
def count_sessions(df):
    if df is None:
        return np.nan
    if "sessions" in df.columns:
        return df["sessions"].sum()
    if "session_id" in df.columns:
        return df["session_id"].nunique()
    if "visits" in df.columns:
        return df["visits"].sum()
    if "user_id" in df.columns:
        return df["user_id"].nunique()
    return len(df)

sessions = count_sessions(web)
orders_count = orders["order_id"].nunique() if orders is not None else 0

site_to_order = (orders_count / sessions * 100) if (sessions and sessions > 0) else np.nan

# Paid orders
if payments is not None:
    if "payment_status" in payments.columns:
        paid_orders = payments[payments["payment_status"].str.lower().isin(["completed","paid","successful","succeeded"])]["order_id"].nunique()
    else:
        paid_orders = payments["order_id"].nunique()
else:
    # Fallback to orders if no payments file
    paid_orders = orders_count

order_to_payment = (paid_orders / orders_count * 100) if orders_count > 0 else np.nan

# Average items per order
if order_items is not None:
    avg_items_per_order = order_items.groupby("order_id").size().mean()
else:
    avg_items_per_order = np.nan

# By-region breakdown
region_summary = None
if geography is not None and orders is not None:
    # Merge orders with geography
    if "zip" in orders.columns and "zip" in geography.columns:
        ord_geo = orders.merge(geography[["zip","region"]], on="zip", how="left")
    else:
        ord_geo = orders.copy()
        ord_geo["region"] = "Unknown"
    
    orders_by_region = ord_geo.groupby("region")["order_id"].nunique().rename("orders")
    region_summary = orders_by_region.to_frame()

    # Note: web_traffic.csv in this case doesn''t have zip, so global conversion is best we can do for comparison
    # but we can still show orders by region.

# Summary
summary = pd.Series({
    "sessions": sessions,
    "orders": orders_count,
    "site_to_order_%": round(site_to_order,2) if not np.isnan(site_to_order) else np.nan,
    "paid_orders": paid_orders,
    "order_to_payment_%": round(order_to_payment,2) if not np.isnan(order_to_payment) else np.nan,
    "avg_items_per_order": round(avg_items_per_order,2) if not np.isnan(avg_items_per_order) else np.nan
}).to_frame("value")

print("\nConversion summary:")
print(summary.T.to_string(index=False))

if region_summary is not None:
    print("\nBy-region orders (top 10):")
    print(region_summary.sort_values("orders", ascending=False).head(10))

# Save outputs
out = Path("output")
out.mkdir(exist_ok=True)
summary.to_csv(out / "conversion_summary.csv")
if region_summary is not None:
    region_summary.to_csv(out / "orders_by_region.csv")

print("\nSaved: output/conversion_summary.csv")
