extends VisualDataflow


# Called when the node enters the scene tree for the first time.
func _ready():
	var node = Panel.new()
	node.set_position(Vector2(getNodePosX(), getNodePosY()))
	node.set_size(Vector2(60,60))
	add_child(node)

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
