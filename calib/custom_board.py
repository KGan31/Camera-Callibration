"""Custom lab ChArUco board: geometry built from the README specifications.

Layout (board frame: origin at the top-left corner of the 9x9 square grid,
x right, y down, Z = 0, metres):
  * 9 x 9 squares of 10 cm, grouped into 3 x 3 blocks of 30 x 30 cm.
  * Corner blocks and the centre block carry a 3 x 3 ChArUco patch, the four
    edge-middle blocks are plain board with a 24 cm circular cut-out.
  * Global checker parity: square (r, c) is dark when (r + c) is even, markers
    (7 cm) sit in the centre of the white (odd) squares of the patches.
  * Marker IDs (DICT_5X5_100), recovered from the lab images, run row-major over
    the white squares of the top patches (0-7), then the bottom patches (8-15),
    then the centre patch (16-19).

Valid corners are the 2x2 interior corners of each patch (4 per patch, 20 in all) and, optionally
(diagonal_junctions=True), the 4 diagonal junctions between patches (24 in all).

The class exposes the same attributes/methods as calib.board.BoardModel
(marker_xy, corners, corner_id, square_corner_ids), so calib.predict can be
reused unchanged. Only chessboard corners that are true X-junctions are kept:
all four surrounding squares must lie inside the grid and every dark square
around the corner must exist (not be in a cut-out block).
"""
import numpy as np, cv2

aruco = cv2.aruco

# README: physical specs (metres). Borders are only the margin around the grid.
SQUARE = 0.10
MARKER = 0.07
CUTOUT_DIAMETER = 0.24
BORDER = dict(top=0.028, bottom=0.032, left=0.030, right=0.029)
N = 9            # squares per side
BLOCK = 3        # squares per block side
CUTOUT_BLOCKS = {(0, 1), (1, 0), (1, 2), (2, 1)}   # (block_row, block_col)


def _has_pattern(r, c):
    return 0 <= r < N and 0 <= c < N and (r // BLOCK, c // BLOCK) not in CUTOUT_BLOCKS


class CustomBoardModel:
    def __init__(self, dict_id=aruco.DICT_5X5_100, diagonal_junctions=False):
        """diagonal_junctions: also use the 4 X-junctions where corner patches touch the centre patch.
        Off by default: they add the longest baseline but are predicted by extrapolation and are
        rejected by the corner-quality gate more often (see the ablation in results/real)."""
        self.diagonal_junctions = diagonal_junctions
        self.nx = self.ny = N
        self.sq = SQUARE
        self.dictionary = aruco.getPredefinedDictionary(dict_id)

        # corner_id -> (X, Y, 0); ids cover every interior grid corner (r-1)*(N-1)+(c-1)
        self.corners = np.array([[c * SQUARE, r * SQUARE, 0.0]
                                 for r in range(1, N) for c in range(1, N)])

        # marker squares: white (odd parity) squares inside a pattern block
        sq = [(r, c) for r in range(N) for c in range(N)
              if _has_pattern(r, c) and (r + c) % 2 == 1]
        group = lambda r: 0 if r < BLOCK else (1 if r >= 2 * BLOCK else 2)
        sq.sort(key=lambda rc: (group(rc[0]), rc[0], rc[1]))
        m = (SQUARE - MARKER) / 2
        self.marker_square = {i: rc for i, rc in enumerate(sq)}
        self.marker_xy = {}
        for i, (r, c) in self.marker_square.items():
            x0, y0 = c * SQUARE + m, r * SQUARE + m
            self.marker_xy[i] = np.array([[x0, y0], [x0 + MARKER, y0],
                                          [x0 + MARKER, y0 + MARKER], [x0, y0 + MARKER]])

    def corner_id(self, c, r):
        """Valid X-junction at grid corner (c, r), else None."""
        if not (1 <= c <= N - 1 and 1 <= r <= N - 1):
            return None
        around = [(r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)]
        if all(_has_pattern(rr, cc) for rr, cc in around if (rr + cc) % 2 == 0):
            return (r - 1) * (N - 1) + (c - 1)
        return None

    def square_corner_ids(self, marker_id):
        """Valid chessboard corner ids this marker can predict.

        The corners of its own square, plus those of its four edge-neighbour (dark) squares. The extra
        ones matter at the four diagonal junctions where a corner patch touches the centre patch:
        those corners belong to a dark square, not to a marker square, and are the longest-baseline
        points on the board. They sit one square (10 cm) beyond the marker, a short extrapolation.
        """
        r, c = self.marker_square[marker_id]
        ids = []
        squares = [(r, c)]
        if self.diagonal_junctions:
            squares += [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for rr, cc in squares:
            if not _has_pattern(rr, cc):
                continue
            for cid in (self.corner_id(x, y) for y in (rr, rr + 1) for x in (cc, cc + 1)):
                if cid is not None and cid not in ids:
                    ids.append(cid)
        return ids
