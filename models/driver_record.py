import pandas as pd

from models import Order
from models import Driver


class DriverRecord:
    def __init__(self):
        self.driver_record_table = pd.DataFrame(
            columns=[
                "driver_id",
                "order_id",
                "order_date",
                # driver location
                "driver_lat",
                "driver_lon",
                # pickup loaction
                "pickup_lat",
                "pickup_lon",
                # drop off loaction
                "dropoff_lat",
                "dropoff_lon",
                # area
                "driver_area",
                "pickup_area",
                "dropoff_area",
                "work_time_minutes_before_order",
                "work_time_minutes_after_order",
                "driver_commision",
                "platform_revenue",
            ]
        )

    def add_driver_record(self, order: Order, driver: Driver):
        driver_record_info = pd.DataFrame(
            [
                {
                    "driver_id": driver.driver_id,
                    "order_id": order.order_id,
                    "order_date": order.datetime,
                    "driver_lat": driver.current_lat,
                    "driver_lon": driver.current_lon,
                    "pickup_lat": order.pickup_lat,
                    "pickup_lon": order.pickup_lon,
                    "dropoff_lat": order.dropoff_lat,
                    "dropoff_lon": order.dropoff_lon,
                    "driver_area": driver.current_area,
                    "pickup_area": order.pickup_area,
                    "dropoff_area": order.dropoff_area,
                    "work_time_minutes_before_order": driver.work_time_minutes,
                    "work_time_minutes": driver.work_time_minutes + order.complete_time,
                    "driver_commission": order.driver_commission,
                    "platform_revenue": order.platform_revenue,
                }
            ]
        )
        print(
            f"Add driver record: Driver {driver.driver_id} has accepted {order.order_id} at {order.datetime}"
        )

        if self.driver_record_table.empty:
            self.driver_record_table = driver_record_info
        else:
            self.driver_record_table = pd.concat(
                [self.driver_record_table, driver_record_info], ignore_index=True
            ).drop_duplicates(subset=["driver_id"], keep="last")
