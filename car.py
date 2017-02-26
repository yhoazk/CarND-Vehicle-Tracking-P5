import numpy as np

class CAR():

    def __init__(self, xy_loc=(0,0), rect_size=150):
        self.xy_loc = xy_loc
        self.rect_size = rect_size
        self.not_found_cntr = 4
        self.fifo_len = 3
        self.xy_th = 100
        self.location_fifo = None
        self.updated = False

    def in_range(self, other_xy):
        """
        Decides if the other object is the same by a threshold of location xy
        :param other_xy:
        :return:
        """
        d_X = self.xy_loc[0] - other_xy[0]
        d_Y = self.xy_loc[1] - other_xy[1]
        dist = np.linalg.norm((self.xy_loc, other_xy))
        print("distance: " + str(d_X))
        return -self.xy_th/2 < d_X < self.xy_th/2

    def add_element(self, new_box):
        centerx = (new_box[1][0] - new_box[0][0]) + new_box[0][0]
        centery = (new_box[1][1] - new_box[0][1]) + new_box[0][1]

        if self.location_fifo is None:
            self.location_fifo = []
        elif self.fifo_len > len(self.location_fifo):
            self.location_fifo.pop(0)

        self.not_found_cntr = 4
        self.location_fifo.append(tuple((int(centerx), int(centery))))
        self.xy_loc = tuple(np.mean(self.location_fifo, axis=0))

    def obj_notFound(self):
        if self.not_found_cntr == 1:
            # the object probably is not longer in the screen
            self.not_found_cntr = 0
            return True
        else:
            # the threshold has not been superpased yet
            self.not_found_cntr -= 1
            return False

    def get_rect(self):
        p1 = (int(self.xy_loc[0]-(1.3*self.rect_size/2)), int(self.xy_loc[1]-(1.3*self.rect_size/2)))
        p2 = (int(self.xy_loc[0]+(0.7*self.rect_size/2)), int(self.xy_loc[1]+(0.7*self.rect_size/2)))
        return (p1,p2)



# class CARS():
