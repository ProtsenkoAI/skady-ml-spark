import vk_api
from ..obtainers.vk_obtainers import VkObtainer # just for typing docs
from tqdm.notebook import tqdm
from requests import exceptions as req_exceptions
from .base_retriever import Retriever
from collections import deque
from .. import exceptions


class ObjectRetriever(Retriever):
    """Gets data for one user/group/etc."""
    def __init__(self, session: vk_api.VkApi, user_id=None, *args, **kwargs):
        self.user_id = user_id
        self.reset_session(session)  
        super().__init__()
    
    def get(self, obtainer: VkObtainer):
       resp = obtainer.request(self.session, self.user_id)
       return obtainer.parse(resp)

    def reset_session(self, new_session):
        self.session = new_session

class PoolRetriever(Retriever):
    def __init__(self, session: vk_api.VkApi, pool_ids, use_tqdm=False):
        self.reset_session(session)

        self.unrequested_ids = deque(pool_ids)
        self.results = {}
        self.responses = {}

        self.off_tqdm = not use_tqdm
        self.max_pool = 25
        super().__init__()

    def reset_session(self, new_session):
        new_session.auth()
        self.session = vk_api.requests_pool.VkRequestsPool(new_session)

    def get(self, obtainer: VkObtainer):
        "Supports working with continue_in_new_session() method from base class"
        id_pool = self.unrequested_ids
        pbar = tqdm(total=len(id_pool), disable=self.off_tqdm)
        while id_pool:
            obj_id = id_pool.popleft()
            self._do_and_save_request(obj_id, obtainer)
            pbar.update(1)
        parsed_results = self._parse_results(self.results, obtainer)
        return parsed_results
                
    def _do_and_save_request(self, obj_id, obtainer):
        try:
            resp = obtainer.request(self.session, obj_id)
            self._add_response(obj_id, resp)
        except exceptions.VkRequestException as err:
            self.log(str(err))
        except exceptions.VkRequestRateException as err:
            self.log(str(err))
            raise err

    def _add_response(self, obj_id, resp):
        self.responses[obj_id] = resp
        force_exec = len(self.unrequested_ids) == 0
        if (len(self.responses) >= self.max_pool) or force_exec:
            batch_results = self._exec(self.responses)
            self.results.update(batch_results)
            self.responses = {}

    def _exec(self, responses):
        try:
            self.session.execute()
        except req_exceptions.ConnectionError:
            self.log("connection error occured while executing")
        return self._results_from_resps(responses)

    def _results_from_resps(self, resps):
        for idx, resp in resps.items():
            if resp.error is False:
                self.results[idx] = resp.result

            elif resp.error["error_code"] == 29:
                raise exceptions.VkRequestRateException()
        return self.results

    def _parse_results(self, results, obtainer):
        parsed_results = {}
        for obj_id, result in results.items():
            try:
                parsed = obtainer.parse(result)
                parsed_results[obj_id] = parsed
            except exceptions.VkParseException as err:
                self.log(str(err))

        return parsed_results


    

# class ObjectsPoolData:
#     """Obtain data for multiple users"""
#     # TODO: need to simply add args to request and parse methods
#     def __init__(self, session: vk_api.VkApi, pool_ids, off_tqdm=True):
#         self.requests_pool = vk_api.requests_pool.VkRequestsPool(session)
#         self.pool_ids = pool_ids
#         self.off_tqdm = off_tqdm

#         self.max_pool_size = 25
#         self._set_savers()

#     def _set_savers(self):
#         self.results = {}
#         self.responses = {}

#     def get(self, obtainer: VkObtainer):
#         self._set_savers()
#         self.obtainer = obtainer
        
#         for idx, user_id in tqdm(enumerate(self.pool_ids), 
#                                  disable=self.off_tqdm,
#                                  total=len(self.pool_ids)):

#             resp = obtainer.request(self.requests_pool, user_id)
#             self._check_exceeded(resp)
#             self.responses[user_id] = resp
#             self._exec_if_need(idx, len(self.pool_ids))
#         parsed_results = self._parsed_from_responses(self.responses, obtainer)

#         return parsed_results

#     def _exec_if_need(self, request_idx, total):
#         is_full_pool = (request_idx + 1) % self.max_pool_size == 0
#         is_last_elem = request_idx == total - 1
#         if is_full_pool or is_last_elem:
#             try:
#                 self.requests_pool.execute()
#             except exceptions.ConnectionError:
#                 print("connection error occured while executing")
#                 pass

#     def _get_resp_results(self, responses):
#         for obj_id, resp in responses.items():
#             if resp.error:
#                 continue
#             try:
#                 self.results[obj_id] = resp.result
#             except RuntimeError:
#                 break
#         return self.results

#     def _parsed_from_responses(self, responses, obtainer):
#         response_results = self._get_resp_results(responses)
#         parsed_results = {}
#         for obj_id, result in response_results.items():
#             parsed = obtainer.parse(result)
#             if not parsed is None:
#                 parsed_results[obj_id] = parsed
#         return parsed_results

#     def get_results_from_saved_responses(self, obtainer):
#         responses = self.responses
#         return self._parsed_from_responses(responses, obtainer)

#     def _check_exceeded(self, resp):
#         if resp.error is False:
#             return 
#         if resp.error["error_code"] == 29:
#                     raise RuntimeError("Exceeded vk requests rate")
        

