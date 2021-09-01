### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f34992e8-be91-11eb-1b3e-ed569dcb4b77
begin
	using Pkg
	Pkg.activate(temp=true)
	Pkg.develop(path=abspath(joinpath(@__DIR__, "../lib/Flux.jl")))
	Pkg.add(Pkg.PackageSpec(name="Metalhead", version="0.5.3"))
	Pkg.add(Pkg.PackageSpec(name="BSON", version="0.3.3"))
	Pkg.add("JuliaHubClient")
	Pkg.add("PlutoUI")
	Pkg.add("ImageIO")
	using Flux, Metalhead, BSON, JuliaHubClient, PlutoUI
	Pkg.add("DataSets")
	Pkg.add("TOML")
	Pkg.develop(path=abspath(joinpath(@__DIR__, "../lib/JuliaHubDataRepos.jl")))
	using DataSets, JuliaHubDataRepos, TOML
	JuliaHubDataRepos.init_datasets_integration()
	pushfirst!(DataSets.PROJECT, DataSets.load_project(TOML.parse("""
 	data_config_version=0

 	[[datasets]]
	name = "resnet_model"
	uuid = "676a52c8-7fa4-4015-aa21-efbee98cb935"
	description = ""

	[datasets.storage]
	driver = "JuliaHubDataRepo"
	bucket_region = "us-east-1"
	bucket = "juliahubprodresults"
	prefix = "datasets"
	version = "v1"
	type = "Blob"
		""")))
	md"Package setup"
end

# ╔═╡ 06cc116d-dffc-4b7e-98a1-e384b1ddd353
auth, _ = JuliaHubClient.authenticate()

# ╔═╡ 269e8b61-5558-4ffa-854a-fb12a6752cf3
jobs = JuliaHubClient.get_jobs(auth = auth)

# ╔═╡ 026acf79-12e8-4e81-bed2-fee1c6a705c9
function html_results_table(name; auth)
    jobs = JuliaHubClient.get_jobs(auth = auth)
    hash = String(rand(['0':'9';'a':'z';'A':'Z'], 10))
    function make_table_row(job)
        time = job.timestamp
        url = JuliaHubClient.get_result_url(job, auth=auth)
        radio = job.status == "Completed" && !isnothing(url) && !isempty(url) ?
            """<input class="_radio" data-url="$(url)"
                type="radio" name="run"></input>""" : ""
        return """
            <tr>
                <td>$(radio)</td>
                <td>$(time)</td>
                <td>$(job.status)</td>
                <td>$(name)</td>
            </tr>"""
    end
    return HTML("""
    <table id="results-table-$hash">
        <thead>
            <td></td>
            <td>Time</td>
            <td>Status</td>
            <td>Name</td>
        </thead>
        <tbody>
            $(
            join((make_table_row(job) for job in jobs if get(job.inputs, "jobname", job.jobname) == name && job.status != "Failed" && job.status != "Stopped"), "\n")
            )
        </tbody>
    </table>

    <script>
        const table = document.getElementById('results-table-$hash')
        const radios = table.getElementsByClassName('_radio')

        for (const r of radios) {
            r.oninput = ev => {
                table.value = r.dataset.url
                table.dispatchEvent(new CustomEvent("input"))
            }
        }
        table.value = null
        table.dispatchEvent(new CustomEvent("input"))
    </script>
    """)
end

# ╔═╡ ba88d69a-6425-4d2a-ad6e-46920136e606
function get_results(url; auth = auth)
	path = tempname()
	if url == nothing || url == ""
		return ""
	end
	JuliaHubClient.get_result_file(url, path, auth = auth)
	if isfile(path)
		return read(path, String)
	else
		return ""
	end
end;

# ╔═╡ e7104637-6c14-4f26-a053-0955bb219083
@bind url html_results_table("resnet_demo", auth = auth)

# ╔═╡ 703ffb58-cbbb-4c57-8c26-63cdeeef22e6
model = url === nothing ? nothing : BSON.load(IOBuffer(get_results(url, auth = auth)), @__MODULE__)[:model]
# model = open(IO, DataSets.dataset("resnet_model")) do io
# 	BSON.load(io, @__MODULE__)[:model]
# end

# ╔═╡ 475d5575-391d-4ab6-8be3-e4d5fd45f30d
fixed_model = model === nothing ? nothing : model[end] isa Dense ? Chain(model..., softmax) : Chain(model[1:end-1]..., softmax);

# ╔═╡ 1a007afb-f207-4123-95ab-9660807a7018
function camera_input(;max_size=342, default_url="https://i.imgur.com/SUmi94P.png")
"""
<span class="pl-image waiting-for-permission">
<style>
	
	.pl-image.popped-out {
		position: fixed;
		top: 0;
		right: 0;
		z-index: 5;
	}

	.pl-image #video-container {
		width: 250px;
	}

	.pl-image video {
		border-radius: 1rem 1rem 0 0;
	}
	.pl-image.waiting-for-permission #video-container {
		display: none;
	}
	.pl-image #prompt {
		display: none;
	}
	.pl-image.waiting-for-permission #prompt {
		width: 250px;
		height: 200px;
		display: grid;
		place-items: center;
		font-family: monospace;
		font-weight: bold;
		text-decoration: underline;
		cursor: pointer;
		border: 5px dashed rgba(0,0,0,.5);
	}

	.pl-image video {
		display: block;
	}
	.pl-image .bar {
		width: inherit;
		display: flex;
		z-index: 6;
	}
	.pl-image .bar#top {
		position: absolute;
		flex-direction: column;
	}
	
	.pl-image .bar#bottom {
		background: black;
		border-radius: 0 0 1rem 1rem;
	}
	.pl-image .bar button {
		flex: 0 0 auto;
		background: rgba(255,255,255,.8);
		border: none;
		width: 2rem;
		height: 2rem;
		border-radius: 100%;
		cursor: pointer;
		z-index: 7;
	}
	.pl-image .bar button#shutter {
		width: 3rem;
		height: 3rem;
		margin: -1.5rem auto .2rem auto;
	}

	.pl-image video.takepicture {
		animation: pictureflash 200ms linear;
	}

	@keyframes pictureflash {
		0% {
			filter: grayscale(1.0) contrast(2.0);
		}

		100% {
			filter: grayscale(0.0) contrast(1.0);
		}
	}
</style>

	<div id="video-container">
		<div id="top" class="bar">
			<button id="stop" title="Stop video">✖</button>
			<button id="pop-out" title="Pop out/pop in">⏏</button>
		</div>
		<video playsinline autoplay></video>
		<div id="bottom" class="bar">
		<button id="shutter" title="Click to take a picture">📷</button>
		</div>
	</div>
		
	<div id="prompt">
		<span>
		Enable webcam
		</span>
	</div>

<script>
	// based on https://github.com/fonsp/printi-static (by the same author)

	const span = currentScript.parentElement
	const video = span.querySelector("video")
	const popout = span.querySelector("button#pop-out")
	const stop = span.querySelector("button#stop")
	const shutter = span.querySelector("button#shutter")
	const prompt = span.querySelector(".pl-image #prompt")

	const maxsize = $(max_size)

	const send_source = (source, src_width, src_height) => {
		const scale = Math.min(1.0, maxsize / src_width, maxsize / src_height)

		const width = Math.floor(src_width * scale)
		const height = Math.floor(src_height * scale)

		const canvas = html`<canvas width=\${width} height=\${height}>`
		const ctx = canvas.getContext("2d")
		ctx.drawImage(source, 0, 0, width, height)

		span.value = {
			width: width,
			height: height,
			data: ctx.getImageData(0, 0, width, height).data,
		}
		span.dispatchEvent(new CustomEvent("input"))
	}
	
	const clear_camera = () => {
		window.stream.getTracks().forEach(s => s.stop());
		video.srcObject = null;

		span.classList.add("waiting-for-permission");
	}

	prompt.onclick = () => {
		navigator.mediaDevices.getUserMedia({
			audio: false,
			video: {
				facingMode: "environment",
			},
		}).then(function(stream) {

			stream.onend = console.log

			window.stream = stream
			video.srcObject = stream
			window.cameraConnected = true
			video.controls = false
			video.play()
			video.controls = false

			span.classList.remove("waiting-for-permission");

		}).catch(function(error) {
			console.log(error)
		});
	}
	stop.onclick = () => {
		clear_camera()
	}
	popout.onclick = () => {
		span.classList.toggle("popped-out")
	}

	shutter.onclick = () => {
		const cl = video.classList
		cl.remove("takepicture")
		void video.offsetHeight
		cl.add("takepicture")
		video.play()
		video.controls = false
		console.log(video)
		send_source(video, video.videoWidth, video.videoHeight)
	}
	
	
	document.addEventListener("visibilitychange", () => {
		if (document.visibilityState != "visible") {
			clear_camera()
		}
	})


	// Set a default image

	const img = html`<img crossOrigin="anonymous">`

	img.onload = () => {
	console.log("helloo")
		send_source(img, img.width, img.height)
	}
	img.src = "$(default_url)"
	console.log(img)
</script>
</span>
""" |> HTML
end


# ╔═╡ 3f7459bf-4d70-4a54-91e2-d68120fcacf3
function process_raw_camera_data(raw_camera_data)
	# the raw image data is a long byte array, we need to transform it into something
	# more "Julian" - something with more _structure_.
	
	# The encoding of the raw byte stream is:
	# every 4 bytes is a single pixel
	# every pixel has 4 values: Red, Green, Blue, Alpha
	# (we ignore alpha for this notebook)
	
	# So to get the red values for each pixel, we take every 4th value, starting at 
	# the 1st:
	reds_flat = UInt8.(raw_camera_data["data"][1:4:end])
	greens_flat = UInt8.(raw_camera_data["data"][2:4:end])
	blues_flat = UInt8.(raw_camera_data["data"][3:4:end])
	
	# but these are still 1-dimensional arrays, nicknamed 'flat' arrays
	# We will 'reshape' this into 2D arrays:
	
	width = raw_camera_data["width"]
	height = raw_camera_data["height"]
	
	# shuffle and flip to get it in the right shape
	reds = reshape(reds_flat, (width, height))' / 255.0f0
	greens = reshape(greens_flat, (width, height))' / 255.0f0
	blues = reshape(blues_flat, (width, height))' / 255.0f0
	
	# we have our 2D array for each color
	# Let's create a single 2D array, where each value contains the R, G and B value of 
	# that pixel
	
	Metalhead.RGB.(reds, greens, blues)
end


# ╔═╡ 5d5470da-edfd-4b4b-8020-8f24612b9cad
@bind cam_data camera_input()

# ╔═╡ 8ec21fb0-a665-4caf-a60b-e5eda2688f12
cam_image = process_raw_camera_data(cam_data)

# ╔═╡ badc679c-44be-47cc-be41-6a599bc424b5
probs = model === nothing ? nothing : fixed_model(Flux.normalize(Metalhead.preprocess(cam_image)));

# ╔═╡ 99938abe-4bae-41fa-ae3b-563efda813ae
model === nothing ? nothing : [probs[i]=>Metalhead.ImageNet.imagenet_labels[i] for i in sortperm(vec(probs))[end:-1:end-2]]

# ╔═╡ Cell order:
# ╟─f34992e8-be91-11eb-1b3e-ed569dcb4b77
# ╟─06cc116d-dffc-4b7e-98a1-e384b1ddd353
# ╟─269e8b61-5558-4ffa-854a-fb12a6752cf3
# ╟─026acf79-12e8-4e81-bed2-fee1c6a705c9
# ╟─ba88d69a-6425-4d2a-ad6e-46920136e606
# ╟─e7104637-6c14-4f26-a053-0955bb219083
# ╟─703ffb58-cbbb-4c57-8c26-63cdeeef22e6
# ╟─475d5575-391d-4ab6-8be3-e4d5fd45f30d
# ╟─1a007afb-f207-4123-95ab-9660807a7018
# ╟─3f7459bf-4d70-4a54-91e2-d68120fcacf3
# ╟─5d5470da-edfd-4b4b-8020-8f24612b9cad
# ╟─8ec21fb0-a665-4caf-a60b-e5eda2688f12
# ╟─badc679c-44be-47cc-be41-6a599bc424b5
# ╟─99938abe-4bae-41fa-ae3b-563efda813ae
