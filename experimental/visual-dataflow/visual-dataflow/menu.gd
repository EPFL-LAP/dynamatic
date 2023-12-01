extends Control

@onready var dotDialog = $FileDialogDot
@onready var csvDialog = $FileDialogCsv


# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_button_dot_pressed():
	dotDialog.show()


func _on_button_csv_pressed():
	csvDialog.show()
