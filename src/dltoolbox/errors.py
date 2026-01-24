class ImageTooSmallError(ValueError):
    """Raised when an image's size is smaller than the required minimum."""

    def __init__(self, image_size: int | tuple[int, int], required_size: int | tuple[int, int]) -> None:
        super().__init__(
            f"image too small: actual={self._format_size(image_size)} < required={self._format_size(required_size)}"
        )
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.required_size = (required_size, required_size) if isinstance(required_size, int) else required_size

    @staticmethod
    def _format_size(size: int | tuple[int, int]) -> str:
        return f"{size}x{size}" if isinstance(size, int) else f"{size[0]}x{size[1]}"
