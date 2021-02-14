class VkException(Exception):
    pass

class VkRequestException(VkException):
    def __init__(self, vk_error, values):
        self.vk_error = vk_error
        self.values = values

    def __str__(self):
        return f"""Error in Vk while trying to send request: \
        {self.vk_error}
        Request values: {self.values}"""

class VkParseException(VkException):
    def __init__(self, message, resp):
        self.message = message
        self.resp = resp

    def __str__(self):
        return f"""Error while parsing vk response occured: \
        {self.message}
        Src response: {self.resp}
        """

class VkRequestRateException(VkException):
    def __str__(self):
        return "Session exceeded requests rate, you should continue with new session"
