import tkinter as tk
import board
import time
import adafruit_dht

class TemperatureMonitorApp:
	def __init__(self, master):
		self.master = master
		master.title("Temperature and Humidity Monitor")
		
		self.temperature_label = tk.Label(master, text="Temperature:")
		self.temperature_label.grid(row=0, column=0)

		self.temperature_value = tk.Label(master, text="")
		self.temperature_value.grid(row=0, column=1)

		self.humidity_label = tk.Label(master, text="Humidity:")
		self.humidity_label.grid(row=1, column=0)

		self.humidity_value = tk.Label(master, text="")
		self.humidity_value.grid(row=1, column=1)

		self.update_values()
	
	def update_values(self):
		dhtDevice = adafruit_dht.DHT11(board.D18)
		try:
			# Print the values to the serial port
			temperature = dhtDevice.temperature
			humidity = dhtDevice.humidity
			print(
				"Temp: {:.1f}°C    Humidity: {}% ".format(
					temperature, humidity
				)
			)

		except RuntimeError as error:
			# Errors happen fairly often, DHT's are hard to read, just keep going
			print(error.args[0])
			time.sleep(2.0)
			
		except Exception as error:
			dhtDevice.exit()
			raise error
			dhtDevice = adafruit_dht.DHT11(board.D18)
			temperature = dhtDevice.temperature
			humidity = dhtDevice.humidity

		if humidity is not None and temperature is not None:
			self.temperature_value.config(text="{:.1f}°C".format(temperature))
			self.humidity_value.config(text="{:.1f}%".format(humidity))
		else:
			self.temperature_value.config(text="Error")
			self.humidity_value.config(text="Error")

		self.master.after(100, self.update_values)

root = tk.Tk()
app = TemperatureMonitorApp(root)
root.mainloop()
