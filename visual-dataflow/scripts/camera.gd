extends Camera2D

const zoom_minimum: Vector2 = Vector2(.5,.5)
const zoom_maximum: Vector2 = Vector2(5.0, 5.0)
const zoom_speed: float = 1.1
var dragging = false
var mouse_start_position
var screen_start_position

var usingSlider = false

# Called when the node enters the scene tree for the first time.
func _ready():
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
	
func zoom_at_point(zoom_change, point):
	var target_zoom = self.zoom * zoom_change
	target_zoom = clamp(target_zoom, zoom_minimum, zoom_maximum)

	# Calculate the position adjustment
	var mouse_pos_in_world = self.get_global_mouse_position()
	var offset = (mouse_pos_in_world - self.global_position) * (1.0 - 1.0 / zoom_change)
	var new_camera_position = self.global_position + offset

	self.zoom = target_zoom
	self.global_position = new_camera_position

func _input(event: InputEvent):
	if !usingSlider:
		if event is InputEventMouseButton:
			if event.is_pressed():
				var mouse_pos = event.position
				if event.button_index == MOUSE_BUTTON_WHEEL_UP:
					if self.zoom < zoom_maximum:
						zoom_at_point(1.2, mouse_pos)
				if event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
					if self.zoom > zoom_minimum:
						zoom_at_point(1/1.2, mouse_pos)
		if event is InputEventMouseButton:
			if event.is_pressed():
				mouse_start_position = event.position
				screen_start_position = self.position
				dragging = true
			else:
				dragging = false
		if event is InputEventMouseMotion and dragging:
			self.position = Vector2(1/self.zoom.x,1/self.zoom.y) * (mouse_start_position - event.position) + screen_start_position

func _on_h_slider_mouse_entered():
	usingSlider = true

func _on_h_slider_mouse_exited():
	usingSlider = false
