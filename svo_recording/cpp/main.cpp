#include <iostream>
#include <csignal>
#include <algorithm>
#include <sl/Camera.hpp>
#include <string>
#include "argparse/argparse.hpp"

using namespace sl;
using namespace std;

struct Settings {
	String output;
	RESOLUTION resolution;
	int fps;
	SVO_COMPRESSION_MODE compression;
};


bool exit_app = false;

void signalHandler(int signum);
Settings* parseArguments(int argc, const char** argv);

int main(int argc, const char** argv) {
	Settings* settings = parseArguments(argc, argv);
	if (settings == nullptr) {
		return -1;
	}

	signal(SIGINT, signalHandler);

	Camera zed;

	InitParameters init_params;
	init_params.camera_resolution = settings->resolution;
	init_params.camera_fps = settings->fps;
	init_params.coordinate_units = UNIT::METER;

	ERROR_CODE err = zed.open(init_params);
	if (err != ERROR_CODE::SUCCESS) {
		cout << err << " exit program " << endl;
		return -1;
	}

	RecordingParameters record_params(settings->output, settings->compression);
	err = zed.enableRecording(record_params);

	if (err != ERROR_CODE::SUCCESS) {
		cout << err << " exit program " << endl;
		return -1;
	}

	int i = 0;

	cout << "Recording until signal is receibed (ctr + c to stop) ..." << endl;
	while (!exit_app) {
		err = zed.grab();
		if (err != ERROR_CODE::SUCCESS) {
			cout << err << " exit program " << endl;
			return -1;
		}

		cout << "Frames recorded: " << i++ << '\r' << flush;
	}

	zed.disableRecording();
	cout << endl << "...End recording" << endl;
	zed.close();
	delete settings;

	return 0;
}

Settings* parseArguments(int argc, const char** argv) {
	ArgumentParser parser;

	parser.addArgument("-o", "--output", 1, false);
	parser.addArgument("-r", "--resolution", 1, false);
	parser.addArgument("-f", "--fps", 1, false);
	parser.addArgument("-c", "--compression", 1, false);

	parser.parse(argc, argv);

	Settings* settings = new Settings;
	settings->output = parser.retrieve<string>("output").c_str();

	string res = parser.retrieve<string>("resolution");
	if (res == "HD2K") {
		settings->resolution = RESOLUTION::HD2K;
	} else if (res == "HD1080") {
		settings->resolution = RESOLUTION::HD1080;
	} else if (res == "HD720") {
		settings->resolution = RESOLUTION::HD720;
	} else if (res == "VGA") {
		settings->resolution = RESOLUTION::VGA;
	} else {
		delete settings;
		parser.usage();
		return nullptr;
	}

	vector<int> fps_vec{15, 30, 60, 100};
	int fps = stoi(parser.retrieve<string>("fps"));
	if (find(fps_vec.begin(), fps_vec.end(), fps) == fps_vec.end()) {
		delete settings;
		parser.usage();
		return nullptr;
	}
	settings->fps = fps;

	string comp = parser.retrieve<string>("compression");
	if (comp == "LOSSLESS") {
		settings->compression = SVO_COMPRESSION_MODE::LOSSLESS;
	} else if (comp == "H264") {
		settings->compression = SVO_COMPRESSION_MODE::H264;
	} else if (comp == "H265") {
		settings->compression = SVO_COMPRESSION_MODE::H265;
	} else {
		delete settings;
		parser.usage();
		return nullptr;
	}

	return settings;
}

void signalHandler(int signum) {
	exit_app = true;
}
