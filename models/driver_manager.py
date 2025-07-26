from .driver import Driver
from .order import Order
from .driver_record import DriverRecord


import pandas as pd
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from geopy.distance import geodesic


class DriverManager:
    """
    Manages the driver information DataFrame.
    """

    def __init__(
        self,
        order_driver_data: pd.DataFrame,
        driver_data: pd.DataFrame,
        schedule_data: pd.DataFrame,
        acceptance_model: BalancedRandomForestClassifier,
        driver_record: DriverRecord,
        sort_driver_pool_policy: str = "distance",
    ):
        self.order_driver_data = order_driver_data
        self.driver_data = driver_data
        self.schedule_data = schedule_data
        # init the update_driver_set
        self.update_driver_set = pd.DataFrame(
            columns=[
                "driver_id",
                "driver_lat",
                "driver_lon",
                "driver_area",
                "work_time_minutes",
            ]
        )
        self.model = acceptance_model
        self.driver_record = driver_record

        if sort_driver_pool_policy == "distance":
            self.sort_driver_pool_policy = "distance"
        elif sort_driver_pool_policy == "random":
            self.sort_driver_pool_policy = "random"
        else:
            self.sort_driver_pool_policy = "distance"
        print(f"The Driver Pool is sorted by {self.sort_driver_pool_policy}")

    def get_original_driver_set(self, order: Order) -> pd.DataFrame:
        """
        Get the original driver set for a specific order.
        """
        original_driver_assign_set = self.order_driver_data[
            self.order_driver_data["order_id"] == order.order_id
        ]
        original_driver_assign_ids = (
            original_driver_assign_set["driver_id"].unique().tolist()
        )

        # select the target driver
        original_driver_set = self.driver_data[
            (self.driver_data["driver_id"].isin(original_driver_assign_ids))
            & (self.driver_data["order_id"] == order.order_id)
        ]

        original_driver_set = original_driver_set[
            [
                "driver_id",
                "driver_lat",
                "driver_lon",
                "driver_area",
                "work_time_minutes",
            ]
        ]
        # original_driver_set = original_driver_set.reset_index(drop=True)
        # Because in the driver data set, if a rider accepts an order, the platform will continue to record the update of his location
        # keep the last record to get the rider's original position for the order
        # original_driver_set = original_driver_set.drop_duplicates(
        #     subset=['driver_id'], keep='last')
        # no need to drop duplicates

        # self.original_driver_set = original_driver_set

        return original_driver_set

    def get_driver_pool(self, order: Order) -> pd.DataFrame:
        """
        Constructs the driver pool, prioritizing updated driver information
        and including new drivers based on matching pickup area.
        """
        original_driver_set = self.get_original_driver_set(order)
        driver_pool = original_driver_set.copy()

        # New Logic: Add drivers from update_driver_set if their area matches order.pickup_area
        # and they are not already in the original driver pool.

        # 1. Filter update_driver_set for drivers matching the pickup area
        area_matched_drivers_from_updates = self.update_driver_set[
            self.update_driver_set["driver_area"] == order.pickup_area
        ].copy()  # Use .copy() to ensure an independent DataFrame

        # 2. Identify truly new drivers (not in original_driver_set) from the area-matched set
        truly_new_drivers_to_add = area_matched_drivers_from_updates[
            ~area_matched_drivers_from_updates["driver_id"].isin(
                driver_pool["driver_id"]
            )
        ]

        # 3. Add these truly new drivers to the driver_pool
        if not truly_new_drivers_to_add.empty:
            driver_pool = pd.concat(
                [driver_pool, truly_new_drivers_to_add], ignore_index=True
            )
            print(
                f"Added new drivers to the pool based on matching 'driver_area' ('{order.pickup_area}'): {truly_new_drivers_to_add['driver_id'].tolist()}"
            )
        else:
            print(
                f"No new drivers from update_driver_set with matching 'driver_area' ('{order.pickup_area}') to add."
            )

        # Prepare for update: Set 'driver_id' as index for both DataFrames.
        # driver_pool now might contain newly added drivers.
        driver_pool_indexed = driver_pool.set_index("driver_id")
        update_set_indexed = self.update_driver_set.set_index("driver_id")

        # Check if driver_pool (which now includes original + potentially new area-matched drivers)
        # and update_set_indexed have common driver_ids.
        # These are the drivers whose information will be updated.
        drivers_to_be_updated = driver_pool_indexed.index.intersection(
            update_set_indexed.index
        )

        # Perform the update operation. This will update rows in driver_pool_indexed
        # where the 'driver_id' exists in update_set_indexed.
        driver_pool_indexed.update(update_set_indexed)

        # Reset the index to 'driver_id' column again for the final DataFrame
        driver_pool = driver_pool_indexed.reset_index()

        # Check if any updates actually happened based on common_driver_ids
        if not drivers_to_be_updated.empty:
            print(
                f"Using data in update_driver_set to update the following drivers in the pool (original and new area-matched): {drivers_to_be_updated.tolist()}"
            )
        else:
            print(
                "update_driver_set has no matched driver ID for existing drivers in the pool, no update from it."
            )

        if self.sort_driver_pool_policy == "random":
            # Randomize the order of the driver_pool
            driver_pool = driver_pool.sample(frac=1).reset_index(drop=True)
            print("Driver pool has been randomized.")
        else:
            # Sorted by distance
            driver_pool["distance"] = driver_pool.apply(
                lambda row: geodesic(
                    (row["driver_lat"], row["driver_lon"]),
                    (order.pickup_lat, order.pickup_lon),
                ).m,
                axis=1,
            )
            driver_pool = driver_pool.sort_values(by="distance", ascending=True)
            print("Driver pool has been sorted by distance.")

        return driver_pool

    def get_driver_attampt(self, order: Order):
        """
        Iterates through the driver pool for a given order.
        You can add your specific logic for each driver inside the loop.
        """
        driver_pool = self.get_driver_pool(order)
        if driver_pool.empty:
            print("Driver pool is empty for this order.")
        for _, driver_info in driver_pool.iterrows():
            driver_id = driver_info["driver_id"]
            driver_lat = driver_info["driver_lat"]
            driver_lon = driver_info["driver_lon"]
            driver_area = driver_info["driver_area"]
            work_time_minutes = driver_info["work_time_minutes"]
            driver = Driver(
                driver_id=driver_id,
                current_lat=driver_lat,
                current_lon=driver_lon,
                current_area=driver_area,
                work_time_minutes=work_time_minutes,
                model=self.model,
            )

            # accept_order = driver.decide_acceptance(
            #     order=order,
            #     schedule_data=self.schedule_data,
            #     threshold=0.5
            # )
            accept_order = driver.decide_acceptance(
                order=order,
                schedule_data=self.schedule_data,
                threshold=np.random.random(),
            )

            if accept_order:
                self.driver_record.add_driver_record(order=order, driver=driver)
                print(f"Order has been accepted, stop driver attampt")
                # New Logic: Update update_driver_set with the accepting driver's info
                accepting_driver_data = pd.DataFrame(
                    [
                        {
                            "driver_id": driver_id,
                            "driver_lat": order.dropoff_lat,  # Use order's pickup latitude
                            "driver_lon": order.dropoff_lon,  # Use order's pickup longitude
                            "driver_area": order.pickup_area,  # Use order's pickup area
                            # Keep driver's current work time
                            "work_time_minutes": work_time_minutes
                            + order.complete_time,
                        }
                    ]
                )

                # Concatenate the existing set with the new driver's data,
                # then drop duplicates to ensure only the latest record for each driver_id is kept.
                # self.update_driver_set = pd.concat(
                #         [self.update_driver_set, accepting_driver_data],
                #         ignore_index=True
                #     ).drop_duplicates(subset=['driver_id'], keep='last')
                # print(f"Updated update_driver_set with driver_id: {driver_id}")

                if self.update_driver_set.empty:
                    self.update_driver_set = accepting_driver_data
                else:
                    self.update_driver_set = pd.concat(
                        [self.update_driver_set, accepting_driver_data],
                        ignore_index=True,
                    ).drop_duplicates(subset=["driver_id"], keep="last")
                print(f"Updated update_driver_set with driver_id: {driver_id}")

                break

        return accept_order
