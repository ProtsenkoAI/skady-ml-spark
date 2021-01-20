#### Documentation for ml api
# Available routes:
###/matching/user_groups
Get user groups by his vk id or short_name 
##### Body arguments:
    user_vk: int (vk id) or str (vk short_name)
    access_token: str with token that gives permission to access user's page info
##### Returns
    user_groups: list of user gorups
    
## /matching/match_values
Returns list of match values with other users
##### Body arguments:
    user_data: json string containing list[int]
    others_data: json string containing list[ list[int] ]
##### Returns
    match_values: list[float]
    
## /matching/sorted_match_vals
Returns indexes of input list sorted by element values
##### Body arguments:
    match_values: list[float] of matches with other users
##### Returns
    sorted_indexes: list[int] with the users' indexes
##### Example
    match_values = [1, 9, 3], returns: [0, 2, 1]