#!pip install pyngrok
#!ngrok authtoken (insert token)

from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8050, bind_tls=True)
public_url
