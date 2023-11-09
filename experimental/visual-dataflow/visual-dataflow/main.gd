extends VisualDataflow

var zoom_minimum = Vector2(.500001,.500001)
var zoom_maximum = Vector2(3.500001,3.500001)
var zoom_speed = 1.10000001

var dragging = false
var mouse_start_position
var screen_start_position

@onready var camera = $Camera2D

func zoom_at_point(zoom_change, point):
	var target_zoom = camera.zoom * zoom_change
	target_zoom = clamp(target_zoom, zoom_minimum, zoom_maximum)

	# Calculate the position adjustment
	var mouse_pos_in_world = camera.get_global_mouse_position()
	var offset = (mouse_pos_in_world - camera.global_position) * (1.0 - 1.0 / zoom_change)
	var new_camera_position = camera.global_position + offset

	camera.zoom = target_zoom
	camera.global_position = new_camera_position

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.is_pressed():
			var mouse_pos = event.position
			if event.button_index == MOUSE_BUTTON_WHEEL_UP:
				if camera.zoom < zoom_maximum:
					zoom_at_point(1.2, mouse_pos)
			if event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				if camera.zoom > zoom_minimum:
					zoom_at_point(1/1.2, mouse_pos)
	if event is InputEventMouseButton:
		if event.is_pressed():
			mouse_start_position = event.position
			screen_start_position = camera.position
			dragging = true
		else:
			dragging = false
	elif event is InputEventMouseMotion and dragging:
		camera.position = Vector2(1/camera.zoom.x,1/camera.zoom.y) * (mouse_start_position - event.position) + screen_start_position

# Called when the node enters the scene tree for the first time.
func _ready():
	print("Ready !")
	addPanel()

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
	

func _on_next_pressed():
	print("Next Cycle")

func _on_prev_pressed():
	print("Previous Cycle")
