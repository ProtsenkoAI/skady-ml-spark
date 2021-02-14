import unittest

from helpers import tests_config, std_objects

config = tests_config.TestsConfig()


class TestRecommender(unittest.TestCase):
    def setUp(self):
        self.user_colname = "user_id"

    def test_get_recommends(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=100)
        some_users = interacts[self.user_colname].unique()
        assistant.update_with_interacts(interacts)

        recommender = std_objects.get_recommender()
        recs = recommender.get_recommends(some_users, assistant)

        all_items = assistant.get_all_items()
        nitems = len(all_items)
        for user_rec in recs:
            for some_item in user_rec[:10]:
                self.assertIn(some_item, all_items)
            self.assertIsInstance(user_rec, list)
            self.assertEqual(len(user_rec), nitems)

    def test_score_items(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=100)
        user = interacts[self.user_colname][0]
        assistant.update_with_interacts(interacts)

        recommender = std_objects.get_recommender()
        user_items_scores = recommender.score_all_items(user, assistant)
        self.assertEqual(len(user_items_scores), len(assistant.get_all_items()))
        self.assertIsInstance(user_items_scores, list)