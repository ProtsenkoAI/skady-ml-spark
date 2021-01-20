# Requests:
### 1: POST /matching/add_user
##### Arguments:
    user_vk: id (int) or shortname (str)
    access_token: str or None. If None, will use default session.
##### Returns:
    user groups
### 2: POST /matching/delete_user
##### Arguments:
    user_id: int

### 3: GET /matching/get_user_matches
* TODO: write args
##### Arguments:
    user_id: user id in system form
##### Returns: 
    List with users' ids