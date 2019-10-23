import 'package:bubble_bottom_bar/bubble_bottom_bar.dart';
import 'dart:convert';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';


class page_server extends StatefulWidget {
  @override
  _page_serverState createState() => _page_serverState();
}

class _page_serverState extends State<page_server> {
  Uint8List buffer_img;
  String download_filename;
  String download_server;
  Image socket_img;

  Future<void> _consocket(BuildContext context) async {
    int port = 9009;
    Socket socket = await Socket.connect("imagenius.iptime.org",port);

    socket.add(utf8.encode('$download_filename'));
    List<int> buffer = [];
    await for (List<int> v in socket.asBroadcastStream()) {
      buffer.addAll(v);
      print(v.length);
      if (v.length == 0) {
        break;
      }
    }

    print("buffer len: "+ buffer.length.toString());
    buffer_img = Uint8List.fromList(buffer);
    this.setState((){
      socket_img = Image.memory(buffer_img);

    });

//    _base64 = Base64Encoder(img);
//    socket_img = await File("assets/server.jpg").writeAsBytes(buffer);
  }

  Widget _sockimg(){
    if(buffer_img == null ){
      return Text("no image selected!");

    } else{
      return socket_img;
    }

  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body:SingleChildScrollView(
        child: Center(
          child: Column(
            children: <Widget>[
              Padding(padding: EdgeInsets.all(10.0)),

              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Flexible(
                    child:TextField(
                      obscureText: false,
                      decoration: InputDecoration(
                        border: OutlineInputBorder(),
                        labelText: 'Download file_name',

                      ),
                      onChanged: (String str) {
                        setState(() => download_filename = str);
                      },
                    ),
                  ),
                  Padding(padding: EdgeInsets.all(5.0)),
//

                  RaisedButton(onPressed: (){
                    _consocket(context);
                  },
                    child: Text("socket"),
                  ),

                ],
              ),

              Padding(padding: EdgeInsets.all(5.0)),
              Container(
                alignment: Alignment.center,
                width: 400,
                height: 400,
//                  child: _decideImageView(),
                child: _sockimg(),
              ),
            ],
          )
        )


      ),

    );
  }
}



