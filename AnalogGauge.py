import cv2
import numpy as np
import math
from typing import Optional, Tuple

class AnalogGauge:
    """
    Class representing an analog gauge to graphically display values.
    """

    def __init__(self,
                 image: np.ndarray,
                 max_value: int = 200,
                 min_value: int = 0,
                 minor_marks: int = 20,
                 units: str = '',
                 arch: int = 180,
                 phase: int = 0) -> None:
        """
        Initializes the AnalogGauge instance.

        Parameters:
            image (np.ndarray): Background image (3-channel uint8 array).
            max_value (int): Maximum value to be displayed.
            min_value (int): Minimum value to be displayed.
            minor_marks (int): Interval for minor marks.
            units (str): Unit of measurement to display.
            arch (int): Angular arch of the gauge.
            phase (int): Initial angular offset.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("The image must be a 3-channel uint8 array.")

        self.base_image = image.copy()
        self.height, self.width, _ = image.shape

        self.max_value = max_value
        self.min_value = min_value
        self.units = units
        self.minor_marks = minor_marks
        self.physvalue = min_value

        # Current gauge value (stored privately)
        self._position: int = 0

        # Geometric configuration of the gauge
        self.start_angle: int = phase
        self.arch: int = arch
        self.end_angle: int = phase + arch
        self.range: int = abs(max_value - min_value)
        self.factor: float = arch / self.range
        self.factor2: float = self.range / arch

        # Center and radius of the gauge
        self.center: Tuple[int, int] = (self.width // 2, self.height // 2)
        self.radius: int = min(self.width, self.height) // 2 - 60

        # Color configuration (BGR format)
        self.scale_color: Tuple[int, int, int] = (200, 200, 200)
        self.needle_color: Tuple[int, int, int] = (0, 0, 255)
        self.text_color: Tuple[int, int, int] = (255, 255, 255)

        self._init_base_image()

    def _init_base_image(self) -> None:
        """Initializes the base image with static elements."""
        # Use the provided image as the background without overwriting it
        self._draw_gauge_arc()
        self._draw_marks_and_labels()
        self._draw_units_label()

    def _draw_gauge_arc(self) -> None:
        """Draws the gauge arc."""
        cv2.ellipse(self.base_image,
                    self.center,
                    (self.radius, self.radius),
                    0,
                    self.start_angle,
                    self.end_angle,
                    self.scale_color,
                    2)

    def _draw_marks_and_labels(self) -> None:
        """Draws the gauge marks and numerical labels."""
        for pos in range(0, self.range + 1, self.minor_marks):
            angle = self.start_angle + pos * self.factor
            radian = math.radians(angle)
            # Calculate mark coordinates
            start_pt = (int(self.center[0] + math.cos(radian) * (self.radius - 10)),
                        int(self.center[1] + math.sin(radian) * (self.radius - 10)))
            end_pt = (int(self.center[0] + math.cos(radian) * self.radius),
                      int(self.center[1] + math.sin(radian) * self.radius))
            cv2.line(self.base_image, start_pt, end_pt, self.scale_color, 2)

            # Numerical label
            label = str(self.min_value + pos)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = int(self.center[0] + math.cos(radian) * (self.radius + 25) - text_width / 2)
            label_y = int(self.center[1] + math.sin(radian) * (self.radius + 25) + text_height / 2)
            cv2.putText(self.base_image,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.text_color,
                        1,
                        cv2.LINE_AA)

    def _draw_units_label(self) -> None:
        """Draws the unit label on the gauge."""
        cv2.putText(self.base_image,
                    self.units,
                    (self.center[0] - 30, self.center[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    self.text_color,
                    2,
                    cv2.LINE_AA)

    @property
    def needle_angle(self) -> int:
        """Returns the current angle of the needle."""
        return self._position - self.start_angle 
    
    @needle_angle.setter
    def needle_angle(self, value: int) -> None:
        """
        Sets the angle of the needle ensuring it stays within the defined limits.
        
        Parameters:
            value (int): New angle for the needle.
        """
        self.physvalue = int(value * self.factor2) + self.min_value
        self._position = value + self.start_angle

    @property
    def needle_position_range(self) -> int:
        """Returns the current position of the needle."""
        return self.physvalue
    
    @needle_position_range.setter
    def needle_position_range(self, value: int) -> None:
        """
        Sets the position of the needle ensuring it stays within the defined limits.
        
        Parameters:
            value (int): New position for the needle.
        """
        self.physvalue = value
        self._position = self.start_angle + value * self.factor

    def update_display(self) -> np.ndarray:
        """
        Draws the dynamic elements (needle and current value) on a copy of the base image.

        Returns:
            np.ndarray: Updated gauge image.
        """
        display_image = self.base_image.copy()

        # Calculate the needle angle
        radian = math.radians(self._position)

        # Calculate the needle end point
        needle_length = self.radius - 30
        needle_end = (int(self.center[0] + math.cos(radian) * needle_length),
                      int(self.center[1] + math.sin(radian) * needle_length))
        cv2.line(display_image, self.center, needle_end, self.needle_color, 3, cv2.LINE_AA)

        # Draw the center of the gauge
        cv2.circle(display_image, self.center, 6, self.needle_color, -1)

        # Display the current value
        value_text = f"{self.physvalue}"
        cv2.putText(display_image,
                    value_text,
                    (self.center[0] - 30, self.center[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.text_color,
                    2,
                    cv2.LINE_AA)

        return display_image

if __name__ == '__main__':
    # Create a background image
    background_image = np.zeros((400, 600, 3), dtype=np.uint8)
    background_image[:] = (30, 30, 30)

    # Create an instance of AnalogGauge
    gauge = AnalogGauge(image=background_image,
                        max_value=200,
                        min_value=0,
                        minor_marks=20,
                        units="km/h",
                        arch=180,
                        phase=180)

    value = 0
    increasing = True

    while True:
        # Update the gauge value and get the updated image
        gauge.needle_position_range = value
        gauge_image = gauge.update_display()
        cv2.imshow("Analog Gauge Demo", gauge_image)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

        # Increment or decrement the gauge value
        if increasing:
            value += 1
            if value >= gauge.max_value:
                increasing = False
        else:
            value -= 1
            if value <= gauge.min_value:
                increasing = True

    cv2.destroyAllWindows()