import unittest


class TestSaving(unittest.TestCase):
    def _assert_parameters_equal(self, model1, model2):
        for (old_param, new_param) in zip(model1.parameters(), model2.parameters()):
            old_param = old_param.detach().numpy()
            new_param = new_param.detach().numpy()
            if old_param.shape == new_param.shape:
                params_equal = (old_param == new_param).all()
                self.assertTrue(params_equal)
