from typing import List, Tuple, Optional
import torch


def merge_candidates(
    all_candidates: List[Tuple[torch.Tensor, torch.Tensor]], scores: Optional[List[torch.Tensor]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """We communicate the loop candidates as Tuples (ii, jj) into a multiprocessing Queue.
    Since the loop detector runs extremely fast, the Queue will likely contain multiple sets when
    its being pulled from. We therefore might have to merge multiple together.
    """
    if len(all_candidates) == 1:
        if scores is not None:
            return all_candidates[0][0], all_candidates[0][1], scores[0]
        else:
            return all_candidates[0][0], all_candidates[0][1]

    all_ii, all_jj, all_scores = [], [], []
    for i, candidates in enumerate(all_candidates):
        all_ii.append(candidates[0])
        all_jj.append(candidates[1])
        if scores is not None:
            all_scores.append(scores[i])

    all_ii, all_jj = torch.cat(all_ii), torch.cat(all_jj)
    if scores is not None:
        scores = torch.cat(scores)
        return all_ii, all_jj, scores
    else:
        return all_ii, all_jj


def tensor_to_tuple(ii: torch.Tensor, jj: torch.Tensor) -> List[Tuple[int, int]]:
    """Given edges in format ii [torch.Tensor], jj [torch.Tensor], convert into
    a list of edges (i, j)"""
    return [(i, j) for i, j in zip(ii.tolist(), jj.tolist())]


def indices_to_tuple(ii: torch.Tensor, jj: torch.Tensor):
    return tuple(map(tuple, zip(ii.tolist(), jj.tolist())))


def tuple_to_tensor(edges: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a list of edges in format (i, j), convert into
    a tuple of tensors ii, jj"""
    if len(edges) > 1:
        ii, jj = zip(*edges)
        return torch.tensor(ii), torch.tensor(jj)
    elif len(edges) == 1:
        return torch.tensor([edges[0][0]]), torch.tensor([edges[0][1]])
    else:
        return None, None


class TrajectorySegment:
    def __init__(self, start, end, parent=None):
        """
        A segment in the trajectory with hierarchical structure

        Args:
            start (int): Start frame of the segment
            end (int): End frame of the segment (-1 for unfinished)
            parent (TrajectorySegment, optional): Parent segment
        """
        self.start = start
        self.end = end
        self.parent = parent
        self.children = []

    def __repr__(self):
        end_str = str(self.end) if self.end != -1 else "unfinished"
        return f"({self.start}, {end_str})"

    def contains_frame(self, frame):
        """Check if this segment contains the given frame"""
        if self.end == -1:
            return self.start <= frame
        return self.start <= frame < self.end

    def contains_range(self, start, end):
        """Check if this segment fully contains the given range"""
        if self.end == -1:
            return self.start <= start
        return self.start <= start and end <= self.end

    def get_length(self):
        """Get the length of this segment"""
        if self.end == -1:
            return float("inf")  # Unfinished segments have infinite length
        return self.end - self.start

    def get_hierarchy_str(self, level=0):
        """Get a string representation of the segment hierarchy"""
        indent = "  " * level
        result = f"{indent}{self}\n"
        for child in self.children:
            result += child.get_hierarchy_str(level + 1)
        return result

    def find_minimal_segment_for_index(self, index, max_depth=float("inf"), current_depth=0):
        """
        Find the smallest segment containing the given index within the subtree

        Args:
            index (int): Frame index to search for
            max_depth (int): Maximum depth to search in the tree
            current_depth (int): Current depth in the recursion

        Returns:
            TrajectorySegment: The smallest segment containing the index, or None
        """
        if not self.contains_frame(index):
            return None

        # If we've reached the maximum depth, return current segment
        if current_depth >= max_depth:
            return self

        # Check if any children contain the index
        for child in self.children:
            if child.contains_frame(index):
                result = child.find_minimal_segment_for_index(index, max_depth, current_depth + 1)
                if result:
                    return result

        # If no children contain the index or we've reached a leaf, return self
        return self


class TrajectorySegmentManager:
    def __init__(self, t_0=0):
        """
        Initialize a trajectory segment manager with a single segment from t_0 to -1 (unfinished)

        Args:
            t_0 (int): Starting timestamp/frame index
        """
        # Initialize with one segment from t_0 to -1 (end not yet reached)
        self.segments = [TrajectorySegment(t_0, -1)]
        # Counter to track the current position / length of the latest segment
        self.current_counter = None

    def insert_edge(self, i, j):
        """
        Insert an edge (loop) connecting frames i and j, and update segments
        while preserving the hierarchical structure.

        Args:
            i (int): First frame of the edge
            j (int): Second frame of the edge

        Returns:
            bool: True if the edge was inserted successfully, False otherwise
        """
        # Ensure i < j for consistency
        if i > j:
            i, j = j, i

        # Find segments containing i and j
        affected_segments = []
        for segment in self.segments:
            if segment.contains_frame(i) or segment.contains_frame(j) or (i <= segment.start and j >= segment.end):
                affected_segments.append(segment)

        # If either frame is not in any segment, return False
        if not any(segment.contains_frame(i) for segment in affected_segments) or not any(
            segment.contains_frame(j) for segment in affected_segments
        ):
            print(f"Frames {i} and/or {j} not found in any segment.")
            return False

        # Create new segments to replace the affected ones
        new_segments = []
        children_to_preserve = []

        # First, collect all children of affected segments that we need to preserve
        for segment in affected_segments:
            children_to_preserve.extend(segment.children)

            # If segment itself is fully within the new edge, add it as a child to preserve
            if i <= segment.start and segment.end <= j and segment.end != -1:
                children_to_preserve.append(segment)

            # If segment partially overlaps with new edge, we need to create subsegments
            # that will be children of the new segments
            elif segment.contains_frame(i) and not segment.contains_frame(j):
                if segment.start < i:
                    # Part before i
                    children_to_preserve.append(TrajectorySegment(segment.start, i, None))
                # Part from i to segment.end
                if segment.end != -1 and i < segment.end:
                    children_to_preserve.append(TrajectorySegment(i, segment.end, None))

            elif segment.contains_frame(j) and not segment.contains_frame(i):
                # Part from segment.start to j
                if segment.start < j:
                    children_to_preserve.append(TrajectorySegment(segment.start, j, None))
                # Part after j
                if segment.end != -1 and j < segment.end:
                    children_to_preserve.append(TrajectorySegment(j, segment.end, None))

            elif segment.contains_frame(i) and segment.contains_frame(j):
                # Segment contains both i and j
                if segment.start < i:
                    # Part before i
                    children_to_preserve.append(TrajectorySegment(segment.start, i, None))
                # Part from i to j (the loop)
                children_to_preserve.append(TrajectorySegment(i, j, None))
                if segment.end != -1 and j < segment.end:
                    # Part after j
                    children_to_preserve.append(TrajectorySegment(j, segment.end, None))

        # Determine the start and end points of all new segments we need to create
        start_points = set([i, j])
        for segment in affected_segments:
            start_points.add(segment.start)
            if segment.end != -1:
                start_points.add(segment.end)
        start_points = sorted(list(start_points))

        # Create new segments
        # First segment: from min start to i
        min_start = min(segment.start for segment in affected_segments)
        if min_start < i:
            pre_segment = TrajectorySegment(min_start, i)
            new_segments.append(pre_segment)

        # Middle segment: from i to j (the loop)
        loop_segment = TrajectorySegment(i, j)
        new_segments.append(loop_segment)

        # Last segment: from j to -1 or max end
        max_end = -1
        for segment in affected_segments:
            if segment.end != -1 and segment.end > j and (max_end == -1 or segment.end > max_end):
                max_end = segment.end

        if max_end == -1 or j < max_end:
            post_segment = TrajectorySegment(j, max_end)
            new_segments.append(post_segment)

        # Assign children to the appropriate new parent segments
        for child in children_to_preserve:
            for potential_parent in new_segments:
                # If the new segment entirely contains the child
                if potential_parent.start <= child.start and (
                    potential_parent.end == -1 or child.end <= potential_parent.end or child.end == -1
                ):
                    child.parent = potential_parent
                    potential_parent.children.append(child)
                    break

        # Remove affected segments from the list
        for segment in affected_segments:
            if segment in self.segments:
                self.segments.remove(segment)

        # Add new segments to the list
        self.segments.extend(new_segments)

        # Sort segments by start time
        self.segments.sort(key=lambda x: x.start)

        return True

    def advance_counter(self, new_counter):
        """
        Advance the current counter and update the last segment's end if needed

        Args:
            new_counter (int): New counter value
        """
        if self.current_counter is not None:
            if new_counter < self.current_counter:
                print(f"Cannot advance counter to {new_counter}. Current counter is {self.current_counter}.")
                return
        self.current_counter = new_counter

    def find_segment_for_index(self, index, max_depth=float("inf")):
        """
        Find the segment containing the given index and return its top-level parent

        Args:
            index (int): Frame index to search for
            max_depth (int): Maximum depth to search in the tree

        Returns:
            tuple: (top_level_segment, minimal_segment) or (None, None) if not found
        """
        # Find the top-level segment containing the index
        top_level_segment = None
        for segment in self.segments:
            if segment.contains_frame(index):
                top_level_segment = segment
                break

        if not top_level_segment:
            return None, None

        # Find the minimal segment within the subtree
        minimal_segment = top_level_segment.find_minimal_segment_for_index(index, max_depth)

        return top_level_segment, minimal_segment

    def get_finite_length(self, segment: TrajectorySegment) -> int:
        length = segment.get_length()
        # In case we have an open segment, simply use the counter to calculate the length
        if length == float("inf") and self.current_counter is not None:
            if self.current_counter < segment.start:
                print(
                    f"""Warning. Current counter {self.current_counter} is less than the start of the segment you are trying to index. 
                    Please update the counter to have a finite length of the open segment ..."""
                )
            else:
                length = self.current_counter - segment.start
        return length

    def get_neighbor_segments(self, index, max_window=float("inf"), num_neighbors=1, max_depth=float("inf")):
        """
        Get segments around the given index, including the segment containing the index
        and its neighbors, constrained by a maximum window size.

        Args:
            index (int): Frame index to search for
            max_window (int): Maximum total length of all segments combined
            num_neighbors (int): Maximum number of neighbor segments on each side
            max_depth (int): Maximum depth to search in the tree

        Returns:
            list: List of segments within the constraints
        """
        # Find the segment containing the index
        top_level_segment, minimal_segment = self.find_segment_for_index(index, max_depth)

        if not minimal_segment:
            return []

        result = [minimal_segment]
        current_length = self.get_finite_length(minimal_segment)  # In case the segment is open,

        # Find all top-level segments
        all_segments = sorted(self.segments, key=lambda x: x.start)

        # Find the index of the top-level segment containing our target
        segment_idx = all_segments.index(top_level_segment)

        # Add neighbors to the left (earlier in time)
        left_count = 0
        left_idx = segment_idx - 1
        while left_count < num_neighbors and left_idx >= 0:
            left_segment = all_segments[left_idx]
            # Check if adding this segment would exceed the max window
            if current_length + self.get_finite_length(left_segment) <= max_window:
                result.insert(0, left_segment)
                current_length += self.get_finite_length(left_segment)
                left_count += 1
            else:
                break
            left_idx -= 1

        # Add neighbors to the right (later in time)
        right_count = 0
        right_idx = segment_idx + 1
        while right_count < num_neighbors and right_idx < len(all_segments):
            right_segment = all_segments[right_idx]
            # Check if adding this segment would exceed the max window
            if current_length + self.get_finite_length(right_segment) <= max_window:
                result.append(right_segment)
                current_length += self.get_finite_length(right_segment)
                right_count += 1
            else:
                break
            right_idx += 1

        # Alternative approach: if we want to find child segments at a certain depth
        # instead of top-level segments, we can search the tree at that depth
        if max_depth > 0 and minimal_segment.children:
            # Get child segments at the desired depth
            child_segments = self._get_child_segments_at_depth(minimal_segment, 1, max_depth)

            # Filter child segments to those containing or adjacent to the index
            relevant_children = []
            for child in sorted(child_segments, key=lambda x: x.start):
                if (
                    child.contains_frame(index)
                    or (len(relevant_children) > 0 and child.start == relevant_children[-1].end)
                    or (not relevant_children and child.end == index)
                    or (child.start == index)
                ):
                    if current_length + self.get_finite_length(child) <= max_window:
                        relevant_children.append(child)
                        current_length += self.get_finite_length(child)

            # Add child segments if we found any
            if relevant_children:
                # Replace the minimal segment with its children in the result
                min_segment_idx = result.index(minimal_segment)
                result = result[:min_segment_idx] + relevant_children + result[min_segment_idx + 1 :]

        return result

    def _get_child_segments_at_depth(self, segment, current_depth, target_depth):
        """Helper function to get child segments at a specific depth"""
        if current_depth == target_depth:
            return [segment]

        result = []
        for child in segment.children:
            result.extend(self._get_child_segments_at_depth(child, current_depth + 1, target_depth))

        return result

    def get_segments(self):
        """
        Get the current list of segments (the top level segments)

        Returns:
            list: List of TrajectorySegment objects
        """
        return self.segments

    def get_minimal_segments(self):
        """
        Get the current low-level child list of segments

        Returns:
            list: List of all TrajectorySegment objects
        """
        all_segments = []
        for segment in self.segments:
            if len(segment.children) > 0:
                for child in segment.children:
                    all_segments.append(child)
            else:
                all_segments.append(segment)
        return all_segments

    def is_complete(self):
        """
        Check if the trajectory is complete (no segments end with -1)

        Returns:
            bool: True if the trajectory is complete, False otherwise
        """
        return not any(segment.end == -1 for segment in self.segments)

    def __str__(self):
        """
        String representation of the segments

        Returns:
            str: String describing the current segments
        """
        return "Trajectory Segments: " + " ".join(str(segment) for segment in self.segments)

    def print_hierarchy(self):
        """
        Print the full hierarchical structure of segments
        """
        result = "Segment Hierarchy:\n"
        for segment in self.segments:
            result += segment.get_hierarchy_str()
        return result


# Example usage
if __name__ == "__main__":
    # Initialize the segment manager
    manager = TrajectorySegmentManager(0)
    print(manager)  # Should show (0, -1)

    # Insert the first edge (3, 10)
    manager.insert_edge(3, 10)
    print(manager)  # Should show (0, 3) (3, 10) (10, -1)

    # Insert another edge (5, 8) which is within an existing segment
    manager.insert_edge(5, 8)
    print(manager)  # Should show (0, 3) (3, 5) (5, 8) (8, 10) (10, -1)

    # Insert an edge that spans multiple segments
    manager.insert_edge(2, 12)
    print(manager)  # Should show (0, 2) (2, 12) (12, -1)
    print(manager.print_hierarchy())

    # Test finding a segment for a specific index
    print("\nTesting find_segment_for_index:")
    top, minimal = manager.find_segment_for_index(6, max_depth=2)
    print(f"For index 6: Top-level segment = {top}, Minimal segment = {minimal}")

    # Test getting neighbor segments with constraints
    print("\nTesting get_neighbor_segments:")
    neighbors = manager.get_neighbor_segments(6, max_window=8, num_neighbors=2, max_depth=2)
    print(f"Neighbor segments for index 6 (max_window=8, num_neighbors=2, max_depth=2):")
    for segment in neighbors:
        print(f"  {segment}")
