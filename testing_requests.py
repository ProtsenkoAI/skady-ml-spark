import requests
import json

access_token = 'af90a3ee8c8a452988fc313a3429fae6aef3704006fd74c382e32e657e0c5e9f3ef9bd05d1e7d9dbf14c4'
ip = "0.0.0.0"

def get_user_groups(user_vk):
    resp = requests.post(f"http://{ip}:5000/matching/user_groups",
                         data={"access_token": access_token, "user_vk": user_vk})
    # print(resp.status_code, type(resp.json()["user_groups"]))
    groups = resp.json()["user_groups"]
    return groups


def get_matches_of_user(user_data, others):
    resp = requests.post(f"http://{ip}:5000/matching/match_values",
                         data={"user_data": json.dumps(user_data), "others_data": json.dumps(others)})
    match_values = resp.json()["match_values"]
    return match_values


def sort_matches_indexes(match_values):
    resp = requests.post(f"http://{ip}:5000/matching/sorted_match_vals",
                         data={"match_values": json.dumps(match_values)})
    return resp.json()["sorted_indexes"]


user1_groups = get_user_groups("egnarts_71")
user2_groups = get_user_groups("markovdigital")
user3_groups = get_user_groups(211471710)

match_val = get_matches_of_user(user1_groups, [user2_groups, user3_groups])
sorted_matches = sort_matches_indexes(match_val)

print(user1_groups)
print(user3_groups)
print(match_val)
print(sorted_matches)