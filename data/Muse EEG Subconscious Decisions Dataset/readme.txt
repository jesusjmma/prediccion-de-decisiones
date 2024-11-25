Data provided on LOCAL folder include decision timing measures in the interface:  
•	ID: identifier of the participant in the experiment. The identifier is an integer, which starts at 0 and has consecutive values. In our case case, it will go up to 19.

•	Trial: session of the experiment in which the results have been recorded. A session runs from the time the "Start Experiment" button is clicked until it appears again. The number of the session is an integer number starting at 0. It is an integer starting at 0 and has consecutive values. In our specific case, it will reach up to 9

•	Response: identifier of each response of each session. A response is from the moment a blank screen is displayed until the letter displayed on the screen is chosen at the moment the impulse to press a key is felt. It is an integer starting at 0 and has consecutive values.

•	Start time: the time at which the response starts, i.e. the time from when a blank screen appears until the decision-making process begins. The decision making process is initiated.

•	Letter appearance time: the time at which random letters start to appear on the screen. From this point onwards, the participant can press the right or left key at any time he/she wishes.

•	Time of the keystroke: the time at which the participant presses the P (right) or Q (left) key.

•	Chosen key: choice made by the participant. It shall have as possible values p and q.

•	Time of appearance of the observed letter: time at which the letter that the participant was asked to.

•	The time of occurrence of the observed letter: time at which the letter appears that the user has been asked to remember at the time he/she makes the free will decision. It is interpreted as the time of the decision.

•	Observed letter: response to the recall of the letter that appeared on the screen at the moment of feeling the free will impulse. It will have as possible values: S, R, N, D, L, C, T, M or #.


--------------------------
--------------------------



The data provided on MUSE folder have the following variables:
 
•	Timestamp: date and time with millisecond precision of the captured data. It is stored in the format YYYYY-MM-DD HH:mm:SS.fff, where YYYYY to the year, MM to the month, DD to the day, HH to the hour, mm to the minute, SS to the second and fff to the millisecond.

•	Delta: brain waves with the largest wave amplitude, mainly active with deep sleep phases, so they are related to processes that do not depend on a state of consciousness. These waves have a frequency of between 1 and 4 Hz.

•	Theta: the brain waves with the largest wave amplitude after theta waves, present in deep calm, relaxation and immersion stages in memories, so they are associated with a present consciousness but disconnected from reality and focused on imaginary experiences. These waves have a frequency of between 4 and 8 Hz.

•	Alpha: the waves with the largest wave amplitude after Theta waves, present in stages of relaxation such as a walk or watching TV, and are therefore related to calm related to processes of deep calm with present awareness. These waves have a frequency of between 7.5 and 13 Hz.

•	Beta: these are the lowest amplitude waves, after gamma waves, present in states that require a certain level of attention or alertness, in which one has to be aware of the changes These waves have a frequency of between 13 and 30 Hz.

•	Gamma: these are the lowest amplitude waves, present in states of wakefulness, which are associated with a broadening of focus and memory management. These waves have a frequency between 30 and 44 Hz.

•	Raw: these are the representation of the raw electrical signals captured by Muse.

•	AUX_RIGHT: raw waveforms captured by an auxiliary USB sensor.

•	Mellow: User relaxation.

•	Concentration: User concentration.

•	Accelerometer (X, Y, Z): detects device movements, tilts, tilts up, tilts down and tilts
•	upwards, downwards and sideways.

•	Gyro (X, Y, Z): gyroscope movement over time.

•	HeadBandOn: indicates if the band is on the head.

•	HSI: sensor quality, the closer to 1 the better the quality.

•	Battery: remaining battery of the device.

•	Elements: different actions that the subject can perform, such as blinking or jaw clenching.




