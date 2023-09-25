extends VisualDataflow


# Called when the node enters the scene tree for the first time.
func _ready():
	print("Loading visual dataflow");
	var x: int = getNumber();
	print("Number is " + str(x));


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
