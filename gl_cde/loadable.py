from abc import ABCMeta, abstractmethod

import torch


class LoadableModelAbstract[TPrepared](metaclass=ABCMeta):
    """Two-phase weight loading: async prepare, then sync materialize on the device thread."""

    @abstractmethod
    async def prepare(self) -> TPrepared:
        """Download and prepare weights.

        Note that this will not materialize the model. Please call materialize with the prepare output.

        Example:
            store = await model.prepare()
            model.materialize(store, torch.device('cuda:0'))

        Returns:
            TPrepared: per-model defined resources required for materializing. Note this does not
                neccesarily contain the weights. Often it's just a temporary filepath or similar.
        """
        ...

    @abstractmethod
    def materialize(self, prepared: TPrepared, device: torch.device) -> None:
        """Initializes the model and loads the weights onto the selected device.

        Args:
            prepared (TPrepared): output from self.prepare()
            device (torch.device): The device to load the weights onto.
        """
        ...
