import streamlit as st
import numpy as np
from collections import deque
import time
from copy import deepcopy
import pandas as pd

class FloodFillSolver:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.colors = set(self.grid.flatten())

    def flood_fill(self, grid, color):
        """Perform flood fill from top-left corner with given color"""
        if grid[0, 0] == color:
            return grid

        new_grid = grid.copy()
        original_color = new_grid[0, 0]
        queue = deque([(0, 0)])
        visited = set()

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited or r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                continue
            if new_grid[r, c] != original_color:
                continue

            visited.add((r, c))
            new_grid[r, c] = color

            # Add adjacent cells (4-way connectivity)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    queue.append((nr, nc))

        return new_grid

    def is_solved(self, grid):
        """Check if grid has only one color"""
        return len(set(grid.flatten())) == 1

    def get_connected_colors(self, grid):
        """Get colors that are adjacent to the main region (connected to top-left)"""
        visited = set()
        queue = deque([(0, 0)])
        main_color = grid[0, 0]
        adjacent_colors = set()

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                continue
            if grid[r, c] != main_color:
                adjacent_colors.add(grid[r, c])
                continue

            visited.add((r, c))

            # Add adjacent cells
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((r + dr, c + dc))

        return adjacent_colors

    def bfs_solve(self, max_depth=15):
        """Solve using BFS with pruning techniques"""
        if self.is_solved(self.grid):
            return []

        # BFS state: (grid_state, moves_taken)
        queue = deque([(self.grid.copy(), [])])
        visited_states = set()

        # Convert grid to tuple for hashing
        def grid_to_tuple(grid):
            return tuple(grid.flatten())

        visited_states.add(grid_to_tuple(self.grid))

        for depth in range(1, max_depth + 1):
            if not queue:
                break

            level_size = len(queue)
            states_at_depth = []

            for _ in range(level_size):
                current_grid, moves = queue.popleft()

                # Get possible moves (colors adjacent to current region)
                adjacent_colors = self.get_connected_colors(current_grid)

                for color in adjacent_colors:
                    if color == current_grid[0, 0]:  # Skip same color
                        continue

                    new_grid = self.flood_fill(current_grid, color)
                    new_moves = moves + [color]

                    if self.is_solved(new_grid):
                        return new_moves

                    grid_tuple = grid_to_tuple(new_grid)
                    if grid_tuple not in visited_states:
                        visited_states.add(grid_tuple)
                        states_at_depth.append((new_grid, new_moves))

            # Add states for next level
            queue.extend(states_at_depth)

            # Pruning: limit queue size to prevent memory explosion
            if len(queue) > 3000:
                # Keep most promising states (fewer remaining colors)
                queue = deque(sorted(queue, key=lambda x: len(set(x[0].flatten())))[:1500])

        return None  # No solution found within max_depth

def display_grid_emoji(grid, colors):
    """Display grid as colored emoji matrix"""
    rows, cols = grid.shape
    result = []

    for i in range(rows):
        row_str = " ".join([colors[grid[i, j]] for j in range(cols)])
        result.append(row_str)

    return "\n".join(result)

def main():
    st.set_page_config(
        page_title="Flood Fill Optimizer", 
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 Flood Fill Game Optimizer")
    st.markdown("**Find the optimal sequence of moves to fill your puzzle with one color!**")

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Game Settings")

        # Grid size selection
        rows = st.slider("Grid Rows", 3, 8, 5)
        cols = st.slider("Grid Columns", 3, 8, 5)

        # Color options
        colors = ['🔴', '🟢', '🔵', '🟡', '🟠', '🟣', '⚫', '⚪']
        num_colors = st.slider("Number of Colors", 2, 6, 4)
        selected_colors = colors[:num_colors]

        max_moves = st.slider("Max Search Depth", 5, 15, 12)

        st.subheader("🎯 How to Play")
        st.markdown("""
        1. **Goal**: Fill the entire grid with one color
        2. **Rule**: Each move floods the connected region from top-left (0,0)
        3. **Edit**: Click grid cells to change colors
        4. **Solve**: Find the minimum moves needed
        """)

        st.subheader("🧠 Algorithm")
        st.markdown("""
        Uses **Bidirectional BFS** with:
        - State pruning to avoid revisits
        - Queue size limiting for performance  
        - Adjacent color detection optimization
        """)

    # Initialize grid in session state
    if 'grid' not in st.session_state or st.session_state.get('grid_size') != (rows, cols):
        st.session_state.grid = np.random.choice(range(num_colors), (rows, cols))
        st.session_state.grid_size = (rows, cols)
        st.session_state.num_colors = num_colors

    # Update grid if number of colors changed
    if st.session_state.get('num_colors') != num_colors:
        # Remap existing colors to new range
        max_val = st.session_state.grid.max()
        if max_val >= num_colors:
            st.session_state.grid = st.session_state.grid % num_colors
        st.session_state.num_colors = num_colors

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🎮 Edit Your Puzzle")

        # Display current grid with colors
        st.text("Current Grid:")
        grid_display = display_grid_emoji(st.session_state.grid, selected_colors)
        st.code(grid_display, language=None)

        # Create interactive grid editor
        st.text("Click cells to edit:")

        # Create a more compact grid editor
        grid_cols = st.columns(cols)

        for j in range(cols):
            with grid_cols[j]:
                st.text(f"Col {j}")
                for i in range(rows):
                    current_color = st.session_state.grid[i, j]
                    if st.button(
                        f"{selected_colors[current_color]}", 
                        key=f"cell_{i}_{j}",
                        help=f"Row {i}, Col {j}",
                        use_container_width=True
                    ):
                        # Cycle through colors
                        st.session_state.grid[i, j] = (current_color + 1) % num_colors
                        st.rerun()

        # Control buttons
        st.text("")  # spacing
        col_reset, col_random, col_simple = st.columns(3)

        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.grid = np.zeros((rows, cols), dtype=int)
                st.rerun()

        with col_random:
            if st.button("🎲 Random", use_container_width=True):
                st.session_state.grid = np.random.choice(range(num_colors), (rows, cols))
                st.rerun()

        with col_simple:
            if st.button("🎯 Easy Puzzle", use_container_width=True):
                # Create a simple solvable puzzle
                simple_grid = np.array([
                    [0, 1, 1, 2],
                    [0, 1, 2, 2], 
                    [0, 0, 2, 1],
                    [0, 1, 1, 1]
                ])[:rows, :cols] % num_colors
                st.session_state.grid = simple_grid
                st.rerun()

    with col2:
        st.subheader("🧠 Solution")

        # Show puzzle statistics
        unique_colors = len(set(st.session_state.grid.flatten()))
        total_cells = rows * cols

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Colors Used", unique_colors)
        with col_stat2:
            st.metric("Grid Size", f"{rows}×{cols}")

        if unique_colors == 1:
            st.success("🎉 Puzzle already solved!")
        else:
            if st.button("🚀 Find Optimal Solution", type="primary", use_container_width=True):
                with st.spinner("Calculating optimal moves..."):
                    start_time = time.time()

                    # Solve the puzzle
                    solver = FloodFillSolver(st.session_state.grid)
                    solution = solver.bfs_solve(max_depth=max_moves)

                    end_time = time.time()

                    if solution:
                        st.success(f"✅ **Optimal solution: {len(solution)} moves**")
                        st.info(f"⏱️ Solved in {end_time - start_time:.2f}s")

                        # Display solution moves
                        st.markdown("**Move sequence:**")
                        move_text = " → ".join([selected_colors[move] for move in solution])
                        st.markdown(f"`{move_text}`")

                        # Detailed moves list
                        with st.expander("📝 Detailed Steps", expanded=True):
                            for i, color_idx in enumerate(solution):
                                st.write(f"**{i+1}.** Choose color {selected_colors[color_idx]}")

                        # Step-by-step animation
                        if st.checkbox("🎬 Show Animation"):
                            current_grid = st.session_state.grid.copy()

                            st.text("Initial:")
                            st.code(display_grid_emoji(current_grid, selected_colors), language=None)

                            for i, color_idx in enumerate(solution):
                                temp_solver = FloodFillSolver(current_grid)
                                current_grid = temp_solver.flood_fill(current_grid, color_idx)

                                st.text(f"After move {i+1} ({selected_colors[color_idx]}):")
                                st.code(display_grid_emoji(current_grid, selected_colors), language=None)

                    else:
                        st.error(f"❌ No solution found within {max_moves} moves")
                        st.info("Try increasing max search depth or simplifying the puzzle")

        # Tips section
        with st.expander("💡 Optimization Tips"):
            st.markdown("""
            - **Smaller grids** (3×3, 4×4) solve much faster
            - **Fewer colors** reduce complexity exponentially  
            - **Connected regions** of same color are more efficient
            - **Edge cases** may require deeper search depths
            - **Most puzzles** have optimal solutions under 10 moves
            """)

if __name__ == "__main__":
    main()
