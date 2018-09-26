# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: oriolrt
'''

class config:
  """
    Configuration class to open a connection to an Oracle server. All the parameters required for a proper connection are stored here
    """

  def __init__(self, params):
    # def __init__(self, IPsHosts,port="1521",username='system',password='oracle',sid='xe'):
    self.username = params["username"]
    if "password" in params:
        self.password = params["password"]
    else:
        self.password = ""

    if not "sid" in params:
      self.sid = "ee"
    else:
      self.sid = params["sid"]

    if "db" in params: self.db = params["db"]

    self.hosts = {}
    for key in params["servers"]:
      serverParams = params["servers"][key]
      self.hosts[int(key)] = serverParams

      if serverParams.has_key("port"):
        self.hosts[int(key)]["port"] = int(serverParams["port"])
      else:
        self.hosts[int(key)]["port"] = 1521