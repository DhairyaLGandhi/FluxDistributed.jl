using Documenter
using ResNetImageNet

DocMeta.setdocmeta!(ResNetImageNet, :DocTestSetup, :(using ResNetImageNet); recursive = true)
makedocs(modules = [ResNetImageNet],
         doctest = VERSION == v"1.6",
         sitename = "Data Parallel Training",
         pages = ["Home" => "index.md",
                  "Training" => "training.md",
                  "Datasets" => "datasets.md"],
         format = Documenter.HTML(
             analytics = "UA-36890222-9",
             assets = ["assets/flux.css"],
             prettyurls = get(ENV, "CI", nothing) == "true"),
         )

deploydocs(repo = "github.com/DhairyaLGandhi/ResNetImageNet.jl.git",
           target = "build",
           devbranch = "main",
           push_preview = true)
