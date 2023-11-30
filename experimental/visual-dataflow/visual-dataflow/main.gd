extends VisualDataflow

@onready var play_button = $CanvasLayer/MarginContainer/VBoxContainer/HBoxContainer/Play
@onready var goToCycle = $CanvasLayer/MarginContainer/VBoxContainer/HBoxContainer/GoToCycle
@onready var legend = $CanvasLayer/GeneralLegend/SpecificLegendsVB

var is_playing = false
var elapsed_time = 0.0
var time_interval = 1.0;

func _ready():
	start()
	
func _process(delta):
	if is_playing:
		elapsed_time += delta
		if elapsed_time >= time_interval:
			nextCycle()
			elapsed_time = 0
	
func _input(event: InputEvent):
	if event.is_action_pressed("ui_right"):
		_on_next_pressed()
	if event.is_action_pressed("ui_left"):
		_on_prev_pressed()
		
func _on_next_pressed():
	nextCycle()

func _on_prev_pressed():
	previousCycle()

func _on_h_slider_value_changed(value):
	changeCycle(value)

func _on_play_pressed():
	if is_playing:
		play_button.text = "Play"
	else:
		play_button.text = "Pause"
	is_playing = !is_playing
	elapsed_time = 0

func _on_go_to_cycle_text_submitted(value):
	changeCycle(int(value))
	goToCycle.clear()


func _on_show_legend_toggled(button_pressed):
	if button_pressed:
		legend.show()
	else:
		legend.hide()
