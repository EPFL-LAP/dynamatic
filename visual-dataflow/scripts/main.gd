extends VisualDataflow

@onready var timeline = $CanvasLayer/Timeline
@onready var play_button = $CanvasLayer/Timeline/MarginContainer/VBoxContainer/HBoxContainer/Play
@onready var goToCycle = $CanvasLayer/Timeline/MarginContainer/VBoxContainer/HBoxContainer/GoToCycle
@onready var legend = $CanvasLayer/Legend
@onready var legendSubView = $CanvasLayer/Legend/GeneralLegend/Panel
@onready var menu = $CanvasLayer/Menu
@onready var drawGraphButton = $CanvasLayer/Menu/VBoxContainer/DrawGraph
@onready var dotInput = $CanvasLayer/Menu/VBoxContainer/ButtonDot
@onready var csvInput = $CanvasLayer/Menu/VBoxContainer/ButtonCsv
@onready var camera = $Camera2D

var is_playing = false
var elapsed_time = 0.0
var time_interval = 1.0;

var dotFile = ""
var csvFile = ""

var color0 = Color(0.8, 0, 0, 1)
var color1 = Color(0, 0, 0, 1)
var color2 = Color(0, 0, 0.8, 1)
var color3 = Color(0, 0.8, 0, 1)
var color4 = Color(0, 0.8, 0.8, 1)

func _ready():
	legend.hide()
	timeline.hide()
	
	for argument in OS.get_cmdline_args():
		if argument.find("=") > -1:
			var key_value = argument.split("=")
			var key = key_value[0].lstrip("--")
			var value = key_value[1]
			if (key == "dot"):
				dotFile = value
			elif (key == "csv"):
				csvFile = value
			else:
				print("Unknown argument " + key)
	if (!dotFile.is_empty() && !csvFile.is_empty()):
		menu.hide()
		legend.show()
		timeline.show()
		start(dotFile, csvFile)

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
	if event is InputEventMouseButton:
		if event.is_action_pressed("left_click"):
			onClick(camera.get_global_mouse_position())


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
		legendSubView.show()
	else:
		legendSubView.hide()

func _on_draw_graph_pressed():
	if (dotFile != "" && csvFile != ""):
		menu.hide()
		legend.show()
		timeline.show()
		start(dotFile, csvFile)

func _on_file_dialog_dot_file_selected(path):
	dotFile = path
	dotInput.text = path

func _on_file_dialog_csv_file_selected(path):
	csvFile = path
	csvInput.text = path


func _on_color_picker_button_popup_closed():
	changeStateColor(0, color0)


func _on_color_picker_button_2_popup_closed():
	changeStateColor(1, color1)


func _on_color_picker_button_3_popup_closed():
	changeStateColor(2, color2)


func _on_color_picker_button_4_popup_closed():
	changeStateColor(3, color3)


func _on_color_picker_button_5_popup_closed():
	changeStateColor(4, color4)


func _on_color_picker_button_color_changed(color):
	color0 = color


func _on_color_picker_button_2_color_changed(color):
	color1 = color


func _on_color_picker_button_3_color_changed(color):
	color2 = color


func _on_color_picker_button_4_color_changed(color):
	color3 = color


func _on_color_picker_button_5_color_changed(color):
	color4 = color


func _on_reset_selection_pressed():
	resetSelection()
