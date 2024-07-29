import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from ..utils.minimum_bounding_box import MinimumBoundingBox

class Edge(object):
    def __init__(self, edge_xyz, edge_uv):
        """Push contact point.
        Args:
            edge_xyz (numpy.ndarray): (N, 3) array of edge points in world frame.
            edge_uv (numpy.ndarray): (N, 2) array of edge points in image coordinates.
        """
        
        self.edge_xyz = edge_xyz
        self.edge_uv = edge_uv
    
    @property
    def center(self):
        return self.edge_xyz.mean(0)

    def visualize_on_image(self, image):
        # get random 1000 point on edge_uv
        rand_idx = np.random.randint(0, len(self.edge_uv), 1000)

        # visualize contact point
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray')
        plt.plot(self.edge_uv[rand_idx, 0], self.edge_uv[rand_idx, 1], 'ko')
        plt.show()

    def visualize_on_cartesian(self):
        # get random 1000 point on edge_xy
        rand_idx = np.random.randint(0, len(self.edge_xyz), 1000)
        
        fig = plt.figure()
        # plot edge points
        plt.plot(self.edge_xyz[rand_idx, 0], self.edge_xyz[rand_idx, 1], 'ko')
        plt.show()   

    @property
    def sampled_edge_xy(self, sample:int=1000):
        # get random 1000 point on edge_xy
        rand_idx = np.random.randint(0, len(self.edge_xyz), sample)
        # plot edge points
        return self.edge_xyz[rand_idx, 0], self.edge_xyz[rand_idx, 1]

    @property
    def min_bbox(self):
        """
        Minimum bounding box in contact local frame.

        Returns:
            min_bbox (numpy.ndarray): (4, 2) minimum bounding box.
        """
        min_bbox = MinimumBoundingBox(self.edge_xyz[:,:2]).corner_points
        min_bbox = np.array(list(min_bbox))
        
        # transfrom min bbox to local frame
        min_bbox = min_bbox - self.pose[:2]
        rot_mat = np.array([
            [np.cos(-self.pose[2]), -np.sin(-self.pose[2])],
            [np.sin(-self.pose[2]), np.cos(-self.pose[2])]])
        min_bbox = np.dot(rot_mat, min_bbox.T)
        min_bbox = min_bbox.T

        # sort as ccw
        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        def convex_hull(points):
            n = len(points)
            l = 0
            for i in range(1,n):
                if points[i][0] < points[l][0]:
                    l = i
            hull = [points[l]]
            p = l
            while True:
                q = (p+1)%n
                for i in range(n):
                    if ccw(points[p], points[i], points[q]):
                        q = i
                p = q
                if p == l:
                    break
                hull.append(points[p])
            return np.array(hull)

        return convex_hull(min_bbox)
        
class EdgeSampler(object):
    '''
    Samples edge points from a masked depth image
    Masked depth image -> Edge points 
    
    '''
    def __init__(self, camera_intr, camera_extr):
        self.camera_intr = camera_intr
        self.camera_extr = camera_extr
        self._width_error_threshold = 1e-3
        
    def sample(self, masked_depth_image):

        # Get point cloud of the object only
        # pcd = self.depth_to_pcd(masked_depth_image, camera_intrinsic)
        # pcd_object = pcd[np.where(pcd[:,2] > 0.1)[0]]
        pcd_object = self.depth_to_pcd(masked_depth_image, self.camera_intr)
        
        # Transform point cloud to world frame
        pcd_w = (np.matmul(self.camera_extr[:3,:3], pcd_object[:,:3].T) + self.camera_extr[:3,3].reshape(3,1)).T
        
        #########################
        #  Height Thresholding ##
        #########################
        
        # threshold_height = 0.01
        ## Remove points that are too close to the ground
        # pcd_w = pcd_w[np.where(pcd_w[:,2] > threshold_height)[0]]
        
        max_height = np.max(pcd_w[:,2]) - (np.max(pcd_w[:,2]) - np.min(pcd_w[:,2])) * 0.1
        pcd_w = pcd_w[np.where(pcd_w[:,2] < max_height)[0]]
        min_height = np.min(pcd_w[:,2]) + (np.max(pcd_w[:,2]) - np.min(pcd_w[:,2])) * 0.05
        pcd_w = pcd_w[np.where(pcd_w[:,2] > min_height)[0]]
        ##########################################
        # Edge Detection - alpha shape algorithm #
        ##########################################
        
        # Calculate the Delaunay triangulation of the point cloud
        pcd_w_2d = pcd_w[:,:2]

        # Define the alpha value (adjust according to your data)
        # alpha_value = 500

        # # Calculate the alpha shape of the point cloud
        # alpha_shape = alphashape.alphashape(pcd_w_2d, alpha=alpha_value)
        
        # if type(alpha_shape) == MultiPolygon:
        #     xs, ys = [], []
        #     for poly in alpha_shape.geoms:
        #         exterior = poly.exterior.coords[:]
        #         x, y = zip(*exterior)
        #         xs += list(x)
        #         ys += list(y)
        #     xs = np.array(xs).reshape(-1, 1)
        #     ys = np.array(ys).reshape(-1, 1)
        #     outermost_points = np.hstack((xs, ys))
        # elif type(alpha_shape) == Polygon:
        #     outermost_points = np.array(alpha_shape.exterior.coords)
        # Get the points on the precise contour
        # outermost_points = np.array(alpha_shape.exterior.coords)
        
        # Find the convex hull of the point cloud
        hull = ConvexHull(pcd_w_2d)

        # Get the indices of the points on the outermost contour
        outermost_indices = hull.vertices
        
        # Get the points on the outermost contour
        outermost_points = pcd_w_2d[outermost_indices]
        
        # Extract x and y coordinates from the contour points
        # x = outermost_points[:, 0]
        # y = outermost_points[:, 1]
        # num_interpolated_points = 500
        # # Create an interpolation function for x and y coordinates separately
        # interpolation_function_x = interp1d(np.arange(len(x)), x, kind='linear')
        # interpolation_function_y = interp1d(np.arange(len(y)), y, kind='linear')

        # # Generate evenly spaced indices for interpolation
        # interpolation_indices = np.linspace(0, len(x)-1, num=num_interpolated_points)

        # # Interpolate x and y coordinates using the interpolation functions
        # x_interpolated = interpolation_function_x(interpolation_indices)
        # y_interpolated = interpolation_function_y(interpolation_indices)

        # # Create the interpolated trajectory with m points (m, 2)
        # interpolated_contour_points = np.column_stack((x_interpolated, y_interpolated))
        # edge_list_xyz = np.hstack([interpolated_contour_points, 0.005 * np.zeros(len(interpolated_contour_points)).reshape(-1,1)]).reshape(-1,3)
        

        num_interpolated_points = 1000
        outermost_indices = np.append(outermost_indices, outermost_indices[0])
        edge_list_xyz = self.interpolate_with_even_distance(pcd_w[outermost_indices], num_interpolated_points)

        # Get uv coordinates of the edge list
        edge_list_xyz_camera = (np.matmul(np.linalg.inv(self.camera_extr)[:3,:3], edge_list_xyz[:,:3].T) + np.linalg.inv(self.camera_extr)[:3,3].reshape(3,1)).T
        edge_list_uvd = edge_list_xyz_camera @ self.camera_intr.T
        edge_list_uv = edge_list_uvd[:,:2] / edge_list_uvd[:,2].reshape(-1,1)
        edge_list_uv = edge_list_uv.astype(int)
        
        return Edge(edge_list_xyz, edge_list_uv)

    @staticmethod
    def remove_outliers(array, threshold=3):
        # Calculate the mean and standard deviation of the array
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)

        # Calculate the Z-scores for each data point
        z_scores = np.abs((array - mean) / std)

        # Filter out the outliers based on the threshold
        filtered_array = array[(z_scores < threshold).all(axis=1)]

        return filtered_array

    @staticmethod
    def interpolate_with_even_distance(trajectory, num_sample):
        '''
        From a trajectory, interpolate the points with even Euclidean distances (xy-plane).
        
        Args:
            trajectory (N,3): Trajectory points
            num_sample (int): Number of points to be sampled
        Returns:
            interpolated_trajectory (num_sample,3): Interpolated trajectory points
        '''
        # Extract the x and y coordinates from the trajectory
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        z = trajectory[:, 2]

        # Compute the cumulative distance along the trajectory
        distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        distances = np.insert(distances, 0, 0)  # Prepend a zero for the initial position

        # Create an interpolation function for x and y coordinates
        interp_func_x = interp1d(distances, x, kind='linear')
        interp_func_y = interp1d(distances, y, kind='linear')
        interp_func_z = interp1d(distances, z, kind='linear')

        # Generate evenly spaced distances for the interpolated points
        target_distances = np.linspace(0, distances[-1], num_sample)

        # Interpolate the x and y coordinates at the target distances
        interpolated_x = interp_func_x(target_distances)
        interpolated_y = interp_func_y(target_distances)
        interpolated_z = interp_func_z(target_distances)

        # Return the interpolated x and y coordinates as a (m, 2) trajectory
        interpolated_trajectory = np.column_stack((interpolated_x, interpolated_y, interpolated_z))
        return interpolated_trajectory
    
    @staticmethod
    def depth_to_pcd(depth_image, camera_intr):
        height, width = depth_image.shape
        row_indices = np.arange(height)
        col_indices = np.arange(width)
        pixel_grid = np.meshgrid(col_indices, row_indices)
        pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
        pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
        depth_arr = np.tile(depth_image.flatten(), [3, 1])
        point_cloud = depth_arr * np.linalg.inv(camera_intr).dot(pixels_homog)
        return point_cloud.transpose()