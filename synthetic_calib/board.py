import numpy as np, cv2
aruco = cv2.aruco

class BoardModel:
    def __init__(self, squares=(10, 7), square_len=0.035, marker_len=0.026,
                 dict_id=aruco.DICT_5X5_100):
        """
        Initialize the board model.

        squares      : tuple of (columns, rows) of the chessboard
        square_len   : length of each square in metres
        marker_len   : length of each marker in metres
        dict_id      : ID of the ArUco dictionary to use
        """

        self.nx, self.ny, self.sq = squares[0], squares[1], square_len
        self.dictionary = aruco.getPredefinedDictionary(dict_id)
        cv_board = aruco.CharucoBoard(squares, square_len, marker_len, self.dictionary)

        # corner_id -> (X, Y, 0) in metres, board frame Map
        self.corners = cv_board.getChessboardCorners()          # (54, 3) 
        # print(f"Chessboard corners: {self.corners}")
        # marker_id -> 4 corners (X, Y) in board frame Map, in order: top-left, top-right, bottom-right, bottom-left
        self.marker_xy = {int(i): np.asarray(o)[:, :2].astype(np.float64)
                          for i, o in zip(cv_board.getIds().flatten(),
                                          cv_board.getObjPoints())}
        # print(f"Marker corners: {self.marker_xy}")

    def corner_id(self, c, r):
        """Inner chessboard corner at grid position (c, r), or None on the border."""

        """
        c, r : 0-based column and row of the inner chessboard corner
        returns corner_id or None if on the border
        """

        if 1 <= c <= self.nx - 1 and 1 <= r <= self.ny - 1:
            return (r - 1) * (self.nx - 1) + (c - 1)
        return None

    def square_corner_ids(self, marker_id):
        """IDs of the chessboard corners on the square that holds this marker."""

        """
        marker_id : ID of the ArUco marker
        returns list of corner_ids (0-based) of the four corners of the square that contains this marker
        """

        cx, cy = self.marker_xy[marker_id].mean(axis=0)
        c0, r0 = int(cx // self.sq), int(cy // self.sq)
        ids = [self.corner_id(c, r) for r in (r0, r0 + 1) for c in (c0, c0 + 1)]
        # print(f"Marker {marker_id} at ({cx:.3f}, {cy:.3f}) -> square ({c0}, {r0}) -> corner IDs {ids}")
        return [i for i in ids if i is not None]