# Wiring (NUCLEO-F446RE)

| Pot | Signal | STM32 pin | Nucleo label | ADC channel | CSV column |
|-----|--------|-----------|--------------|-------------|------------|
| P1  | R_L    | PA0       | A0           | IN0         | 1          |
| P2  | X_L    | PA1       | A1           | IN1         | 2          |
| P3  | V1     | PA4       | A2           | IN4         | 3          |
| P4  | V2     | PB0       | A3           | IN8         | 4          |

Each pot: leg 1 -> 3V3, leg 3 -> GND, wiper -> pin above.
Each wiper also gets a 100 nF capacitor to GND.
Never connect pots to 5V.

Serial: USART2 (PA2 TX / PA3 RX) over ST-LINK USB, 115200 8N1.
Output: "<adc_RL>,<adc_XL>,<adc_V1>,<adc_V2>\n" at 20 Hz (12-bit raw counts).
Command: send "PING" or "ID" + newline -> board replies "ID:STM32F446RE-M1".
LED LD2 (PA5): blinks about 1 Hz = firmware running.