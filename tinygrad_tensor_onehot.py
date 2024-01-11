# Functionality: One hot encoding of a tensor using the tinygrad library

from typing import List, Optional, Union

import numpy as np

# from tinygrad.tinygrad.dtype import dtypes
from tinygrad.tinygrad.tensor import Tensor


class ViewableTensor:
    def __init__(self, tensor: Tensor):
        self.shape = tensor.shape
        self.underlying_shape = self.shape
        # Flatten the tensor
        self.tensor = tensor.flatten().contiguous()
        # Init the slice patchwork to be the entire tensor
        start = 0
        stop = self.shape[0]
        self.slice_patchwork = [slice(start, stop, 1)]

    # Function to be able to print the tensor in slices [[slice0start:slice0stop], [slice1start:slice1stop], ...]
    def __str__(self):
        tensors = []
        for slice_ in self.slice_patchwork:
            tensors.append(self.tensor[slice_].data().tolist())
        return str(tensors)

    # UNTESTED: Function to get a slice of the tensor view
    def __getitem__(self, slice_: slice) -> Tensor:
        # Empty slice
        if slice_.start == slice_.stop:
            return Tensor.zeros((0,), device=self.tensor.device, requires_grad=False)
        slice_length = slice_.stop - slice_.start
        slice_step = slice_.step if slice_.step is not None else 1
        # Create a zero tensor of the slice length
        tensor_slice = Tensor.zeros(
            (slice_length,), device=self.tensor.device, requires_grad=False
        )
        # Iterate over the slice patchwork
        index = 0
        for slice_patch in self.slice_patchwork:
            # Slice length of the slice patch
            slice_patch_length = (slice_patch.stop - slice_patch.start) // (
                slice_patch.step if slice_patch.step is not None else 1
            )
            # If the slice_patch_length is less than the remaining slice length then copy the entire slice patch
            if (slice_patch_length // slice_step) <= (slice_length - index):
                tensor_slice[index : (index + slice_patch_length)].assign(
                    self.tensor[slice(slice_patch.start, slice_patch.stop, slice_step)]
                )
                index += slice_patch_length
            # If the slice_patch_length is greater than the remaining slice length then copy the remaining slice length
            else:
                tensor_slice[index : (index + slice_length)].assign(
                    self.tensor[
                        slice(
                            slice_patch.start,
                            slice_patch.start + ((slice_length - index) * slice_step),
                            slice_step,
                        )
                    ]
                )
                break

    # UNTESTED: Function to set a slice of the tensor view
    def __setitem__(self, slice_: slice, tensor: Tensor) -> None:
        # Empty slice
        if slice_.start == slice_.stop:
            return
        slice_length = slice_.stop - slice_.start
        slice_step = slice_.step if slice_.step is not None else 1
        # Iterate over the slice patchwork
        index = 0
        for slice_patch in self.slice_patchwork:
            # Slice length of the slice patch
            slice_patch_length = (slice_patch.stop - slice_patch.start) // (
                slice_patch.step if slice_patch.step is not None else 1
            )
            # If the slice_patch_length is less than the remaining slice length then copy the entire slice patch
            if (slice_patch_length // slice_step) <= (slice_length - index):
                self.tensor[
                    slice(slice_patch.start, slice_patch.stop, slice_step)
                ].assign(tensor[index : (index + slice_patch_length)])
                index += slice_patch_length
            # If the slice_patch_length is greater than the remaining slice length then copy the remaining slice length
            else:
                self.tensor[
                    slice(
                        slice_patch.start,
                        slice_patch.start + ((slice_length - index) * slice_step),
                        slice_step,
                    )
                ].assign(tensor[index : (index + slice_length)])
                break

    def view(
        self, slice_patchwork: List[slice], shape: tuple, safe: Optional[bool] = True
    ):
        if safe:
            # Assert that the slice patchwork is valid by checking that the sum of the
            # lengths of the slices is equal to the length of the tensor
            # Assert that all slice indices are within the bounds of the tensor
            # Not asserting that the slice patchwork is not overlapping
            len_self_tensor_underlying = 1
            for dim in self.underlying_shape:
                assert dim > 0
                len_self_tensor_underlying *= dim
            # Modify the view of the tensor
            self.slice_patchwork = slice_patchwork
            # Modify the shape property of the tensor
            self.shape = shape
            length_new_tensor = 0
            for slice_ in slice_patchwork:
                # Skip the slice if it is empty
                if slice_.start == slice_.stop:
                    continue
                assert slice_.start >= 0
                assert slice_.stop <= len_self_tensor_underlying
                assert slice_.start <= slice_.stop
                slice_length = slice_.stop - slice_.start
                if slice_.step is not None:
                    assert slice_.step > 0
                    assert slice_.step <= slice_length
                    assert slice_length % slice_.step == 0
                    slice_length = slice_length // slice_.step
                length_new_tensor += slice_length
            length_from_shape = 1
            for dim in shape:
                assert dim > 0
                length_from_shape *= dim
            assert length_new_tensor == length_from_shape

    def continuos(self) -> Tensor:
        # Create a new tensor from the slice patchwork
        len_self_tensor = 1
        for dim in self.shape:
            assert dim > 0
            len_self_tensor *= dim
        continuos_view_flat = (len_self_tensor,)
        continuos_view = Tensor.zeros(
            continuos_view_flat, device=self.tensor.device, requires_grad=False
        )
        next_start = 0
        # Iterate over the slice patchwork
        for slice_ in self.slice_patchwork:
            # Get the slice indices
            start = slice_.start
            stop = slice_.stop
            step = slice_.step if slice_.step is not None else 1
            # Get the slice of the tensor
            tensor_slice = self.tensor[start:stop:step]
            len_tensor_slice = (stop - start) // step
            # Get the length of the slice
            slice_length = len_tensor_slice
            # Get the slice indices of the continuos view
            continuos_view_slice = continuos_view[next_start:slice_length]
            # Assign the slice of the tensor to the slice of the continuos view
            continuos_view_slice.assign(tensor_slice)
            next_start += slice_length
        # Reshape if the new shape is not flat
        if self.shape != continuos_view_flat:
            continuos_view = continuos_view.reshape(self.shape)
        # Return the continuos view
        return continuos_view

    def merge(
        self,
        new_shape: tuple,
        other: Optional["ViewableTensor"] = None,
        safe: Optional[bool] = True,
    ) -> Tensor:
        len_from_shape = 1
        for dim in new_shape:
            assert dim > 0
            len_from_shape *= dim
        if safe:
            assert len(self.underlying_shape) == len(other.underlying_shape)
            len_self_tensor = 1
            for dim in self.underlying_shape:
                assert dim > 0
                len_self_tensor *= dim
            len_other_tensor = 1
            for dim in other.underlying_shape:
                assert dim > 0
                len_other_tensor *= dim
            assert len_from_shape == len_self_tensor == len_other_tensor
            # Assert the device of the tensors are the same
            assert self.tensor.device == other.tensor.device
        # Create a new tensor from the slice patchwork
        # Numpy array since tensor is trash
        merged_view = np.zeros((len_from_shape,), dtype=np.uint8)
        # merged_view = Tensor.zeros((len_from_shape,), device=self.tensor.device, requires_grad=False)
        # Iterate over the slice patchwork
        for slice_ in self.slice_patchwork:
            # Get the slice indices
            start = slice_.start
            stop = slice_.stop
            step = slice_.step if slice_.step is not None else 1
            # Get the slice of the tensor
            tensor_slice = self.tensor[start:stop:step]
            # Get the slice indices of the merged view that correspond to the slice of the tensor
            # Assign the slice of the tensor to the slice of the merged view
            merged_view[start:stop:step] = tensor_slice.numpy()

        if other is not None:
            # Iterate over the slice patchwork of the other tensor and ensure that the there is no overlap
            for slice_ in other.slice_patchwork:
                # Get the slice indices
                start = slice_.start
                stop = slice_.stop
                step = slice_.step if slice_.step is not None else 1
                # Get the slice of the tensor
                tensor_slice = other.tensor[start:stop:step]
                # Get the slice indices of the merged view that correspond to the slice of the tensor
                # Assign the slice of the tensor to the slice of the merged view
                merged_view[start:stop:step] = tensor_slice.numpy()
        # Reshape to the new shape
        merged_view = merged_view.reshape(new_shape)
        # Convert back to trash tensor class
        merged_view = Tensor(
            merged_view, device=self.tensor.device, requires_grad=False
        )
        # Return the merged view
        return merged_view


def one_hot(indices: Union[Tensor, List], num_classes: int) -> Tensor:
    # indices: tensor of indices to be one hot encoded
    # num_classes: number of classes to be one hot encoded
    # returns: one hot encoded tensor
    if isinstance(indices, Tensor):
        len_indices = indices.flatten().shape[0]
        # Convert the indices to a list
        indices = indices.data().tolist()
    else:
        len_indices = len(indices)

    # Create a viewable tensor of one row of the one hot encoding but either all zeros or all ones
    ones = ViewableTensor(Tensor.ones((len_indices,), requires_grad=False))
    zeros = ViewableTensor(Tensor.zeros((len_indices,), requires_grad=False))

    one_hot_tensor_rows = []

    for row in range(num_classes):
        # Change the view of the viewable tensor to be the one hot encoding of the index
        ones.view([slice(row, (row + 1), 1)], (1,))
        zeros.view(
            [slice(0, row, 1), slice((row + 1), len_indices, 1)], ((len_indices - 1),)
        )
        # Merge the viewable tensors to create the one hot encoding
        one_hot_tensor_row = zeros.merge((len_indices,), other=ones)
        # Append the one hot encoded row to the one hot encoded tensor
        one_hot_tensor_rows.append(one_hot_tensor_row)

    # Stack the one hot encoded rows to create the one hot encoded tensor
    one_hot_tensor = Tensor.stack(one_hot_tensor_rows)

    return one_hot_tensor


# Test the one hot encoding functionality
if __name__ == "__main__":
    indices = Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], requires_grad=False)
    num_classes = 10
    one_hot_tensor = one_hot(indices, num_classes)
    print("one_hot_tensor.device: ", one_hot_tensor.device)
    one_hot_tensor = one_hot_tensor.numpy()
    print("one_hot_tensor: ", one_hot_tensor)
    print("one_hot_tensor.sum(): ", one_hot_tensor.sum())
    print("one_hot_tensor.shape: ", one_hot_tensor.shape)

    # Assert that the one hot tensor is correct
    assert one_hot_tensor.sum() == num_classes
    assert one_hot_tensor.shape == (num_classes, num_classes)
    assert one_hot_tensor[0][0] == 1
    assert one_hot_tensor[0][1] == 0
    assert one_hot_tensor[1][0] == 0
    assert one_hot_tensor[1][1] == 1
