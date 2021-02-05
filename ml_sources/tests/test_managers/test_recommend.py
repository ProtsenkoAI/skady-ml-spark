import unittest


class TestRecommender(unittest.TestCase):
    def test_get_recommends_with_assistant(self):

        some_users = self.std_interacts[self.user_colname].unique()[:10]
        recommends = self.std_assistant.get_recommends(some_users)
        self.assertEqual(len(recommends), len(some_users))
        # test that every item is in recommends
        self.assertEqual(self.std_model.get_init_kwargs()["nitems"], len(recommends[0]))

    def test_get_items_probs(self):
        some_users = self.std_interacts[self.user_colname].unique()[:10]

        items_probs = self.std_assistant.get_items_probs(some_users)
        self.assertEqual(len(items_probs), len(some_users))
        self.assertEqual(self.std_model.get_init_kwargs()["nitems"], len(items_probs[0]))
