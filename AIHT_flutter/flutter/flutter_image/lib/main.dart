import 'package:countdown/countdown.dart';
import 'dart:io';
import 'package:image/image.dart' as im;
import 'package:bubble_bottom_bar/bubble_bottom_bar.dart';
import 'dart:convert';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:web_socket_channel/io.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:async_loader/async_loader.dart';
import 'page_home.dart';
import 'page_server.dart';
void main() => runApp(MaterialApp(
    title: "camera app",
    home: LandingScreen()

));


class LandingScreen extends StatefulWidget {
  @override
  _LandingScreenState createState() => _LandingScreenState();
}

class _LandingScreenState extends State<LandingScreen> {

  bool isLoading = false;
  Uint8List buffer_img;
  bool conversion = false;
  File imageFile;
  File conv_img;
  String download_filename;
  String download_server;
  Image socket_img;


  int currentIndex;
  final List<Widget> _children = [
    page_home(),
    page_server()

  ];


  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    currentIndex = 0;

  }

  void changePage(int index) {
    setState(() {
      currentIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('My Flutter App'),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {},
        child: Icon(Icons.add),
        backgroundColor: Colors.red,
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endDocked,
      bottomNavigationBar: BubbleBottomBar(
        hasNotch: true,
        fabLocation: BubbleBottomBarFabLocation.end,
        opacity: .2,
        currentIndex: currentIndex,
        onTap: changePage,
        borderRadius: BorderRadius.vertical(
            top: Radius.circular(
                16)), //border radius doesn't work when the notch is enabled.
        elevation: 8,
        items: <BubbleBottomBarItem>[
          BubbleBottomBarItem(
              backgroundColor: Colors.red,
              icon: Icon(
                Icons.dashboard,
                color: Colors.black,
              ),
              activeIcon: Icon(
                Icons.dashboard,
                color: Colors.red,
              ),
              title: Text("Home")),
          BubbleBottomBarItem(
              backgroundColor: Colors.deepPurple,
              icon: Icon(
                Icons.file_download,
                color: Colors.black,
              ),
              activeIcon: Icon(
                Icons.file_download,
                color: Colors.deepPurple,
              ),
              title: Text("download")),

        ],
      ),
      body: _children[currentIndex], // new

    );
  }


}

