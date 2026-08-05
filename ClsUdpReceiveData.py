# ライブラリ等のインポート ---------------------------------------
import numpy as np
import socket
import os
if os.name != 'nt':
	import netifaces

class ClsUdpReceiveData:
	# コンストラクタ --------------------------------------------
	def __init__(self):
		self.sock_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


	# デストラクタ ----------------------------------------------
	def __del__(self):
		self.sock_receiver.close()


	# ソケットクローズ ------------------------------------------
	def close(self):
		self.sock_receiver.close()


	# 自己IPアドレス取得 ----------------------------------------
	def check_own_ip(self, str_interface):
		if os.name == 'nt':
			str_host = socket.gethostname()
			str_own_ip = socket.gethostbyname(str_host)
		else:
			str_own_ip = netifaces.ifaddresses(str_interface)[netifaces.AF_INET][0]['addr']
		
		return str_own_ip


	# 自己IPアドレス設定 ----------------------------------------
	def bind_socket(self, str_own_ip, sc_own_port):
		self.sock_receiver.bind((str_own_ip, sc_own_port))
	

	# タイムアウト設定 ------------------------------------------
	def set_timeout(self, sc_timeout = 0):
		self.sock_receiver.settimeout(sc_timeout) # time out in sc_timeout second


	# パケット長設定 --------------------------------------------
	def set_data_length(self, sc_header_length, sc_payload_length, sc_trailer_length):
		self.sc_header_length = sc_header_length
		self.sc_payload_length = sc_payload_length
		self.sc_trailer_length = sc_trailer_length
		self.sc_data_length = sc_header_length + sc_payload_length + sc_trailer_length


	# データ受信 -----------------------------------------------
	def receive_data(self):
		try:
			ve_receive_buffer, adr_sender = self.sock_receiver.recvfrom(self.sc_data_length)
			sc_payload_end = self.sc_header_length + self.sc_payload_length
			ve_header = np.frombuffer(ve_receive_buffer[0:self.sc_header_length], dtype='uint8')
			ve_payload = np.frombuffer(ve_receive_buffer[self.sc_header_length:sc_payload_end], dtype='uint8')
			ve_trailer = np.frombuffer(ve_receive_buffer[sc_payload_end:self.sc_data_length], dtype='uint8')
		except TimeoutError:
			print("Timed out")
			ve_header = None
			ve_payload = None
			ve_trailer = None

		return ve_header, ve_payload, ve_trailer


if __name__ == '__main__':
	import numpy as np
	import os
	import cv2
	from ClsUdpReceiveData import ClsUdpReceiveData

	sc_port_number = 50002
	sc_header_length = 0
	sc_payload_length = 19200
	sc_trailer_length = 4
	sc_num_of_image = -1
	sc_max_num_of_image = 8
	bl_buffer_ready = False
	him_received = np.uint8(np.zeros((sc_max_num_of_image, sc_payload_length)))

	receiver = ClsUdpReceiveData()
	receiver.bind_socket("127.0.0.1", sc_port_number)
	receiver.set_data_length(sc_header_length, sc_payload_length, sc_trailer_length)
	receiver.set_timeout(5)

	while True:
		ve_header_rx, ve_payload_rx, ve_trailer_rx = receiver.receive_data()

		if sc_num_of_image == -1:
			if ve_trailer_rx[0] == 0:
				sc_num_of_image = ve_trailer_rx[1]
		else:
			sc_image_number = ve_trailer_rx[0] 
			him_received[sc_image_number,:] = ve_payload_rx

			if sc_image_number == sc_num_of_image - 1:
				for i in range(sc_num_of_image):
					im_reshaped = cv2.resize(him_received[i,:].reshape(
						(120,160)), (320, 240))
					imstr = "Image " + str(i)
					cv2.imshow(imstr, im_reshaped)
			
				sc_key = cv2.waitKey(1) & 0xFF
				if sc_key == ord('q'):
					break
