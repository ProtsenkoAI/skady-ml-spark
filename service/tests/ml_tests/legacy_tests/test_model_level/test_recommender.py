import unittest

from helpers import tests_config, std_objects

config = tests_config.TestsConfig()


class TestRecommender(unittest.TestCase):
    def test_get_recommends(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=100)
        some_users = interacts[config.user_colname].unique()
        items = interacts[config.item_colname].unique()
        assistant.update_with_interacts(interacts)

        recommender = std_objects.get_recommender()
        recs = recommender.get_recommends(some_users, assistant, items)

        all_items = interacts[config.item_colname].unique()
        nitems = len(all_items)
        for user_rec in recs:
            for some_item in user_rec[:10]:
                self.assertIn(some_item, all_items)
            self.assertIsInstance(user_rec, list)
            self.assertEqual(len(user_rec), nitems)

    def test_score_items(self):
        assistant = std_objects.get_assistant()
        interacts = std_objects.get_interacts(nrows=100)
        user = interacts[config.user_colname][0]
        all_items = interacts[config.item_colname].unique()
        assistant.update_with_interacts(interacts)

        recommender = std_objects.get_recommender()
        user_items_scores = recommender.score_all_items(user, assistant, all_items)
        self.assertEqual(len(user_items_scores), len(all_items))
        self.assertIsInstance(user_items_scores, list)
