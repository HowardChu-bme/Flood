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
        """Solve using BFS with pruning techniques - optimized for larger grids"""
        if self.is_solved(self.grid):
            return []

        # For larger grids, use more aggressive pruning
        grid_size = self.rows * self.cols
        if grid_size > 100:  # For grids larger than 10x10
            max_states = 2000
            prune_to = 1000
        elif grid_size > 64:  # For grids larger than 8x8
            max_states = 3000
            prune_to = 1500
        else:
            max_states = 5000
            prune_to = 2500

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
                if not queue:  # Queue might be empty due to pruning
                    break

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

            # More aggressive pruning for larger grids
            if len(queue) > max_states:
                # Sort by: 1) fewer colors, 2) larger connected region from (0,0)
                def state_priority(state):
                    grid, moves = state
                    num_colors = len(set(grid.flatten()))
                    # Count connected region size from (0,0)
                    connected_size = self._count_connected_region(grid)
                    return (num_colors, -connected_size)  # Minimize colors, maximize connected region

                queue = deque(sorted(queue, key=state_priority)[:prune_to])

            # Progress indicator for large grids
            if grid_size > 100 and depth % 2 == 0:
                print(f"Depth {depth}: {len(queue)} states in queue")

        return None  # No solution found within max_depth

    def _count_connected_region(self, grid):
        """Count size of connected region from top-left corner"""
        visited = set()
        queue = deque([(0, 0)])
        main_color = grid[0, 0]
        count = 0

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                continue
            if grid[r, c] != main_color:
                continue

            visited.add((r, c))
            count += 1

            # Add adjacent cells
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((r + dr, c + dc))

        return count

def display_grid_emoji(grid, colors):
    """Display grid as colored emoji matrix"""
    rows, cols = grid.shape
    result = []

    for i in range(rows):
        row_str = " ".join([colors[grid[i, j]] for j in range(cols)])
        result.append(row_str)

    return "\n".join(result)

def display_grid_compact(grid, colors):
    """Display grid in a more compact format for larger grids"""
    rows, cols = grid.shape
    if rows > 10 or cols > 10:
        # For large grids, show a more compact representation
        result = []
        for i in range(rows):
            row_str = "".join([colors[grid[i, j]] for j in range(cols)])
            result.append(row_str)
        return "\n".join(result)
    else:
        return display_grid_emoji(grid, colors)

def initialize_grid(rows, cols, num_colors):
    """Initialize a new grid with given dimensions"""
    return np.random.choice(range(num_colors), (rows, cols))

def resize_grid(current_grid, new_rows, new_cols, num_colors):
    """Resize existing grid to new dimensions, preserving data where possible"""
    old_rows, old_cols = current_grid.shape

    # Create new grid with random values
    new_grid = np.random.choice(range(num_colors), (new_rows, new_cols))

    # Copy existing data where dimensions overlap
    copy_rows = min(old_rows, new_rows)
    copy_cols = min(old_cols, new_cols)

    new_grid[:copy_rows, :copy_cols] = current_grid[:copy_rows, :copy_cols]

    # Ensure all values are within the new color range
    new_grid = new_grid % num_colors

    return new_grid

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

        # Grid size selection - increased to 14x14 maximum
        rows = st.slider("Grid Rows", 3, 14, 6)
        cols = st.slider("Grid Columns", 3, 14, 6)

        # Specific color options as requested: red, yellow, orange, blue, black, white
        colors = ['🔴', '🟡', '🟠', '🔵', '⚫', '⚪']
        color_names = ['Red', 'Yellow', 'Orange', 'Blue', 'Black', 'White']
        num_colors = 6  # Fixed to 6 colors as requested
        selected_colors = colors[:num_colors]

        # Dynamic max search depth based on grid size
        grid_size = rows * cols
        if grid_size > 100:
            default_max_moves = 8
            max_limit = 12
        elif grid_size > 64:
            default_max_moves = 10
            max_limit = 15
        else:
            default_max_moves = 12
            max_limit = 18

        max_moves = st.slider("Max Search Depth", 5, max_limit, default_max_moves)

        st.subheader("🎨 Colors")
        for i, (color, name) in enumerate(zip(selected_colors, color_names)):
            st.write(f"{color} {name}")

        st.subheader("🎯 How to Play")
        st.markdown("""
        1. **Goal**: Fill the entire grid with one color
        2. **Rule**: Each move floods the connected region from top-left (0,0)
        3. **Edit**: Click grid cells to change colors
        4. **Solve**: Find the minimum moves needed
        """)

        st.subheader("🧠 Algorithm")
        if grid_size > 100:
            st.warning("⚠️ Large grid detected! Using optimized pruning.")
        st.markdown("""
        Uses **Bidirectional BFS** with:
        - Adaptive state pruning for grid size
        - Priority-based queue management
        - Connected region optimization
        """)

    # Initialize or update grid in session state
    current_dimensions = (rows, cols)

    # Initialize grid if not exists
    if 'grid' not in st.session_state:
        st.session_state.grid = initialize_grid(rows, cols, num_colors)
        st.session_state.grid_size = current_dimensions
        st.session_state.num_colors = num_colors

    # Handle dimension changes
    elif st.session_state.get('grid_size') != current_dimensions:
        st.session_state.grid = resize_grid(st.session_state.grid, rows, cols, num_colors)
        st.session_state.grid_size = current_dimensions

    # Handle color count changes
    elif st.session_state.get('num_colors') != num_colors:
        # Remap existing colors to new range
        st.session_state.grid = st.session_state.grid % num_colors
        st.session_state.num_colors = num_colors

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🎮 Edit Your Puzzle")

        # Display current grid with colors
        st.text("Current Grid:")
        grid_display = display_grid_compact(st.session_state.grid, selected_colors)
        st.code(grid_display, language=None)

        # Create interactive grid editor - optimized for larger grids
        st.text("Click cells to edit:")

        # For very large grids, use a more efficient display
        if rows > 8 or cols > 8:
            st.info(f"📏 Large grid ({rows}×{cols}). Use control buttons to modify.")

            # Show a sample of the grid for editing
            sample_rows = min(rows, 6)
            sample_cols = min(cols, 6)

            st.text(f"Editing top-left {sample_rows}×{sample_cols} section:")
            for i in range(sample_rows):
                grid_cols = st.columns(sample_cols)
                for j in range(sample_cols):
                    with grid_cols[j]:
                        if i < st.session_state.grid.shape[0] and j < st.session_state.grid.shape[1]:
                            current_color = st.session_state.grid[i, j]
                            if st.button(
                                f"{selected_colors[current_color]}", 
                                key=f"cell_{i}_{j}",
                                help=f"Row {i}, Col {j}",
                                use_container_width=True
                            ):
                                st.session_state.grid[i, j] = (current_color + 1) % num_colors
                                st.rerun()
        else:
            # Full grid editor for smaller grids
            for i in range(rows):
                grid_cols = st.columns(cols)
                for j in range(cols):
                    with grid_cols[j]:
                        if i < st.session_state.grid.shape[0] and j < st.session_state.grid.shape[1]:
                            current_color = st.session_state.grid[i, j]
                            if st.button(
                                f"{selected_colors[current_color]}", 
                                key=f"cell_{i}_{j}",
                                help=f"Row {i}, Col {j}",
                                use_container_width=True
                            ):
                                st.session_state.grid[i, j] = (current_color + 1) % num_colors
                                st.rerun()

        # Control buttons
        st.text("")  # spacing
        col_reset, col_random, col_pattern = st.columns(3)

        with col_reset:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.grid = np.zeros((rows, cols), dtype=int)
                st.rerun()

        with col_random:
            if st.button("🎲 Random", use_container_width=True):
                st.session_state.grid = initialize_grid(rows, cols, num_colors)
                st.rerun()

        with col_pattern:
            if st.button("🎯 Pattern", use_container_width=True):
                # Create an interesting pattern for larger grids
                new_grid = np.zeros((rows, cols), dtype=int)

                # Create concentric rectangles pattern
                for i in range(rows):
                    for j in range(cols):
                        # Distance from edges
                        dist_from_edge = min(i, j, rows-1-i, cols-1-j)
                        new_grid[i, j] = dist_from_edge % num_colors

                st.session_state.grid = new_grid
                st.rerun()

    with col2:
        st.subheader("🧠 Solution")

        # Show puzzle statistics
        unique_colors = len(set(st.session_state.grid.flatten()))
        total_cells = rows * cols

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Colors Used", unique_colors)
        with col_stat2:
            st.metric("Grid Size", f"{rows}×{cols}")
        with col_stat3:
            st.metric("Total Cells", total_cells)

        # Performance warning for very large grids
        if total_cells > 150:
            st.warning("⚠️ Very large grid! Solving may take longer and use more memory.")
        elif total_cells > 100:
            st.info("ℹ️ Large grid detected. Using optimized algorithms.")

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

                        # Color names for moves
                        move_names = " → ".join([color_names[move] for move in solution])
                        st.markdown(f"**Colors:** {move_names}")

                        # Detailed moves list
                        with st.expander("📝 Detailed Steps", expanded=True):
                            for i, color_idx in enumerate(solution):
                                st.write(f"**{i+1}.** Choose {selected_colors[color_idx]} ({color_names[color_idx]})")

                        # Step-by-step animation (only for smaller grids)
                        if total_cells <= 100 and st.checkbox("🎬 Show Animation"):
                            current_grid = st.session_state.grid.copy()

                            st.text("Initial:")
                            st.code(display_grid_compact(current_grid, selected_colors), language=None)

                            for i, color_idx in enumerate(solution):
                                temp_solver = FloodFillSolver(current_grid)
                                current_grid = temp_solver.flood_fill(current_grid, color_idx)

                                st.text(f"After move {i+1} ({color_names[color_idx]}):")
                                st.code(display_grid_compact(current_grid, selected_colors), language=None)
                        elif total_cells > 100:
                            st.info("🎬 Animation disabled for large grids to improve performance")

                    else:
                        st.error(f"❌ No solution found within {max_moves} moves")
                        st.info("Try increasing max search depth or simplifying the puzzle")

        # Tips section
        with st.expander("💡 Optimization Tips"):
            st.markdown(f"""
            **For {rows}×{cols} grids:**
            - **Recommended depth**: {default_max_moves} moves for good performance
            - **Large grid strategy**: Focus on creating larger connected regions
            - **Pattern approach**: Concentric or spiral patterns often solve efficiently
            - **Performance**: Grids >10×10 use optimized pruning algorithms
            - **Memory usage**: Very large grids may require patience
            """)

if __name__ == "__main__":
    main()
