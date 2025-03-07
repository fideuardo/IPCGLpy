import cv2
import numpy as np

class Arrow:
    def __init__(self, image):
        if len(image.shape) == 2:
            self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            self.image = image.copy()

    def draw_arrow(self, start_point, end_point, tip_length=0.2, color=(0, 255, 0), 
                   thickness=2, solid_arrowhead=True, shaft_cap="round"):
        """
        Draws an arrow on the image.
        
        Args:
            start_point (tuple): (x, y) start coordinate.
            end_point (tuple): (x, y) end coordinate.
            tip_length (float): Relative length of the arrowhead (as a fraction of total arrow length).
            color (tuple): BGR color of the arrow.
            thickness (int): Thickness of the arrow.
            solid_arrowhead (bool): If True, draws a custom solid arrowhead; 
                                    if False, uses cv2.arrowedLine() for a hollow arrowhead.
            shaft_cap (str): 'round' or 'square' cap style for the shaft (only used for solid arrowhead).
        
        For a solid arrowhead, the arrowhead length is computed as:
            arrowhead_length = int(tip_length * arrow_length)
        The shaft is drawn from the start point to the base of the arrowhead.
        """
        arrow_vec = np.array(end_point) - np.array(start_point)
        arrow_length = np.linalg.norm(arrow_vec)
        arrowhead_length = max(1, int(tip_length * arrow_length))
        
        if solid_arrowhead:
            # Calculate the angle of the arrow.
            angle = np.arctan2(arrow_vec[1], arrow_vec[0])
            
            # Compute side points (p1 and p2) for the arrowhead triangle.
            p1 = (int(end_point[0] - arrowhead_length * np.cos(angle - np.pi / 6)),
                  int(end_point[1] - arrowhead_length * np.sin(angle - np.pi / 6)))
            p2 = (int(end_point[0] - arrowhead_length * np.cos(angle + np.pi / 6)),
                  int(end_point[1] - arrowhead_length * np.sin(angle + np.pi / 6)))
            
            # Compute the base center (midpoint between p1 and p2) where the shaft should end.
            base_center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            
            # Draw the shaft according to the desired cap style.
            if shaft_cap.lower() == "round":
                # Draw a line and then add circles at the endpoints.
                cv2.line(self.image, start_point, base_center, color, thickness, cv2.LINE_AA)
                cv2.circle(self.image, start_point, thickness // 2, color, -1)
                cv2.circle(self.image, base_center, thickness // 2, color, -1)
            elif shaft_cap.lower() == "square":
                # Draw the shaft as a rotated rectangle with square ends.
                dx = base_center[0] - start_point[0]
                dy = base_center[1] - start_point[1]
                L = np.hypot(dx, dy)
                if L != 0:
                    udx, udy = dx / L, dy / L
                    # Compute the perpendicular vector.
                    pdx, pdy = -udy, udx
                    half_thick = thickness / 2.0
                    pt1 = (int(start_point[0] + pdx * half_thick), int(start_point[1] + pdy * half_thick))
                    pt2 = (int(start_point[0] - pdx * half_thick), int(start_point[1] - pdy * half_thick))
                    pt3 = (int(base_center[0] - pdx * half_thick), int(base_center[1] - pdy * half_thick))
                    pt4 = (int(base_center[0] + pdx * half_thick), int(base_center[1] + pdy * half_thick))
                    pts = np.array([pt1, pt2, pt3, pt4], np.int32)
                    cv2.fillPoly(self.image, [pts], color)
                else:
                    cv2.line(self.image, start_point, base_center, color, thickness, cv2.LINE_AA)
            else:
                # Default to a simple line if an unrecognized option is provided.
                cv2.line(self.image, start_point, base_center, color, thickness, cv2.LINE_AA)
            
            # Draw the solid arrowhead as a filled triangle.
            points = np.array([end_point, p1, p2], np.int32)
            cv2.fillPoly(self.image, [points], color)
        else:
            # For a hollow arrowhead, use OpenCV's arrowedLine with the provided tip_length.
            cv2.arrowedLine(self.image, start_point, end_point, color, thickness, tipLength=tip_length)

    def get_image(self):
        return self.image

# Example usage:
if __name__ == "__main__":
    image = np.ones((150, 300, 3), dtype=np.uint8) * 255
    arrow_drawer = ArrowDrawer(image)

    # Solid arrow with round shaft cap.
    arrow_drawer.draw_arrow((250, 50), (30, 50), tip_length=0.2, color=(0, 0, 255), thickness=8, 
                              solid_arrowhead=True, shaft_cap="round")

    # Solid arrow with square shaft cap.
    arrow_drawer.draw_arrow((100, 80), (150, 80), tip_length=0.6, color=(0, 255, 0), thickness=10, 
                              solid_arrowhead=True, shaft_cap="square")

    # Hollow arrow using cv2.arrowedLine().
    arrow_drawer.draw_arrow((30, 110), (250, 110), tip_length=0.2, color=(255, 0, 0), thickness=8, 
                              solid_arrowhead=False)

    cv2.imshow("Arrows", arrow_drawer.get_image())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
