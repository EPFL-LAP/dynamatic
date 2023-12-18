extends Node

@onready var panel = $GeneralLegend/Panel


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_show_legend_toggled(button_pressed):
	if button_pressed:
		panel.show()
	else:
		panel.hide()
