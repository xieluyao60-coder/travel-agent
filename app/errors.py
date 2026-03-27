class ServiceError(RuntimeError):
    """Base error for business services."""


class UserInputError(ServiceError):
    """Raised when user input is invalid or incomplete."""


class ExternalAPIError(ServiceError):
    """Raised when an upstream API fails or returns invalid data."""


class WeatherForecastRangeError(UserInputError):
    """Raised when target forecast date is outside weather provider range."""

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(f"当前仅支持查询 {start_date} 到 {end_date} 的天气预报")
