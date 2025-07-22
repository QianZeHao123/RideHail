from datetime import datetime
from .weather import WeatherService


class Order:
    """
    Represents a single customer order with details about pickup, dropoff, pricing,
    and calculated commission/revenue.
    """

    def __init__(
        self,
        order_id: int,
        datetime_str: str,
        pickup_area: int,
        dropoff_area: int,
        pickup_lat: float,
        pickup_lon: float,
        dropoff_lat: float,
        dropoff_lon: float,
        customer_price: float,
        commissionPercent: float,
        complete_time: float,
        weather_service: WeatherService,
    ):
        """
        Initializes an Order object.

        Args:
            order_id (int): Unique identifier for the order.
            datetime_str (str): Date and time of the order creation in '%Y-%m-%d %H:%M:%S.%f' format.
            pickup_area (int): Identifier for the pickup geographical area.
            dropoff_area (int): Identifier for the dropoff geographical area.
            pickup_lat (float): Latitude coordinate of the pickup location.
            pickup_lon (float): Longitude coordinate of the pickup location.
            dropoff_lat (float): Latitude coordinate of the dropoff location.
            dropoff_lon (float): Longitude coordinate of the dropoff location.
            customer_price (float): The total price paid by the customer for the order.
            commissionPercent (float): The percentage of the customer price taken as platform commission (e.g., 0.20 for 20%).
        """
        self.order_id = order_id
        # Convert datetime string to a datetime object for easier manipulation
        # self.datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        # self.datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        self.datetime: datetime = (
            datetime_str.to_pydatetime()
        )  # Convert Timestamp to datetime.datetime object

        self.pickup_area = pickup_area
        self.dropoff_area = dropoff_area
        self.pickup_lat = pickup_lat
        self.pickup_lon = pickup_lon
        self.dropoff_lat = dropoff_lat
        self.dropoff_lon = dropoff_lon
        self.customer_price = customer_price
        self.commissionPercent = commissionPercent
        self.complete_time = complete_time
        # These calculations were previously in __post_init__
        # Calculate the driver's earnings from the order
        # self.driver_commission = self.customer_price * (1 - self.commissionPercent)
        self.driver_commission = self.customer_price * self.commissionPercent
        # Calculate the platform's revenue from the order
        # self.platform_revenue = self.customer_price * self.commissionPercent
        self.platform_revenue = self.customer_price * \
            (1 - self.commissionPercent)
        # Extract the hour of the day when the order was placed (0-23)
        self.hour_of_day = self.datetime.hour
        self.weather_code = weather_service.get_weather_code(self.datetime)

    def __repr__(self):
        """
        Returns a string representation of the Order object for easy debugging and display.
        """
        return (
            f"Order(\n"
            f"    order_id={self.order_id},\n"
            f"    datetime={self.datetime},\n"
            f"    pickup_area={self.pickup_area},\n"
            f"    dropoff_area={self.dropoff_area},\n"
            f"    pickup_lat={self.pickup_lat},\n"
            f"    pickup_lon={self.pickup_lon},\n"
            f"    dropoff_lat={self.dropoff_lat},\n"
            f"    dropoff_lon={self.dropoff_lon},\n"
            f"    customer_price={self.customer_price:.2f},\n"
            f"    commissionPercent={self.commissionPercent:.2f},\n"
            f"    driver_commission={self.driver_commission:.2f},\n"
            f"    platform_revenue={self.platform_revenue:.2f},\n"
            f"    hour_of_day={self.hour_of_day}\n"
            f"    weather_code={self.weather_code}\n"
            f"    complete_time={self.complete_time}\n"
            f")"
        )
