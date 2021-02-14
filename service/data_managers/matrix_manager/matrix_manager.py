class MatrixDataManager:
    def add_row_and_col(self, arr):
        """
        Appends arr as new row AND column to matrix in mongo
        Example:
            matrix = [[0, 0.1],
                      [0.2, 0]]
        add_row_and_col([0.4, 0.5]) -> [[0, 0.1, 0.4],
                                        [0.2, 0, 0.5],
                                        [0.4, 0.5, 0]] ATTENTION: 0 at every diagonal elem
        """
        ...

    def delete_row_and_col(self, idx):
        """
        Reverse operation for add_row_and_col(). Deletes row and col by index
        """
        ...

    def get_row(self, user_id):
        """Returns all match values with other users using user_id
        Returns: """
        ...