# Firmware build notes
1. Open the .ioc in STM32CubeIDE (board: NUCLEO-F446RE, clock 84 MHz default).
2. Build, then Run to flash over the ST-LINK USB.
3. Terminal: 115200 baud. Expect ~20 lines/s of 4 comma-separated numbers.
4. TIM2: PSC=8399, ARR=499 -> 20 Hz.