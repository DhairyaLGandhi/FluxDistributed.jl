struct COCODataset{T}
  dataset_path::T
  labels_path::T
  img_paths
  annotations
end

function COCODataset(task = "instances", dataset = "train")
  dataset_path = joinpath(path"images", "$(dataset)2017")
  labels_path = joinpath(path"annotations", "$(task)_$(dataset)2017_pixelmaps")
  r, anns = open(BlobTree, DataSets.dataset("coco_cyclops")) do data_tree
    readdir(data_tree[dataset_path]), annotations(data_tree, joinpath(path"annotations", string(task, "_", dataset, "2017.json")))
  end
  COCODataset(dataset_path, labels_path, r, anns)
end

function Base.show(io::IO, coco::COCODataset)
  task = first(split(coco.labels_path.components[2], "_"))
  dataset = first(split(coco.dataset_path.components[2], "_"))
  println(io, "COCODataset(", "\"$task\"", ", ", "\"$dataset\"", ")")
end

labels(data_tree, labels = "train") = annotations(data_tree, joinpath(path"annotations", "stuff_$(labels)2017.json"))

function annotations(data_tree, labels_file::DataSets.RelPath = path"annotations/stuff_train2017.json")
  lines = open(IO, data_tree[labels_file]) do io
    JSON3.read(io)
  end
end

function get_categories(data_tree, cats = nothing, dataset = path"annotations/instances_train2017.json")
  anns = annotations(data_tree, dataset)
  all_cats = anns[:categories]
  isnothing(cats) && return all_cats
  filter(x -> any(in(cats), Ref(x[:supercategory])) || any(in(cats), Ref(x[:name])), all_cats)
end

function get_imagepaths(data_tree, dataset = path"images/train2017")
  r = readdir(data_tree[dataset])
  map(r_ -> joinpath(dataset, r_), r)
end

function load_img(r, dataset = path"images/train2017")
  open(IO, data_tree[joinpath(dataset, r)]) do io
    FileIO.load(io)
  end
end
