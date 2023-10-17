extends VisualDataflow


# Called when the node enters the scene tree for the first time.
func _ready():
	print("Loading visual dataflow");
	var x: int = getNumber();
	print("Number is " + str(x));
	var y: float = get_amplitude();
	print("Amplitude is " + str(y));
	set_amplitude(300.0);
	var z: float = get_amplitude();
	print("Amplitude is " + str(z));
	var node = Panel.new();
	node.set_position(Vector2(100, 100));
	node.set_size(Vector2(20,20));
	add_child(node);
	


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	super.my_process(delta)
