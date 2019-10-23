import 'package:bubble_bottom_bar/bubble_bottom_bar.dart';
import 'dart:convert';
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:dio/dio.dart';


class page_home extends StatefulWidget {
  @override
  _page_homeState createState() => _page_homeState();
}

class _page_homeState extends State<page_home> {
  final ipv4 = "localhost:5000";
  bool isLoading = false;
  bool conversion = false;
  File imageFile;
  File conv_img;
  Dio dio = new Dio();
  FormData formdata = new FormData();


  _openGallary(BuildContext context) async{
    var picture = await ImagePicker.pickImage(source: ImageSource.gallery);
    this.setState((){
      imageFile = picture;

    });
    Navigator.of(context).pop();
  }
  _openCamera(BuildContext context) async{
    var picture = await ImagePicker.pickImage(source: ImageSource.camera);
    this.setState((){
      imageFile = picture;

    });
    Navigator.of(context).pop();
  }
  _conversion(BuildContext context) async {
    this.setState(() {
      if (conversion != false){conv_img = imageFile;}
    });
  }


  Future<void> _showChoiceDialog(BuildContext context){
    return showDialog(context: context, builder: (BuildContext context){
      return AlertDialog(
        title: Text("make a choice!"),
        content: SingleChildScrollView(
          child: ListBody(
            children: <Widget>[
              GestureDetector(
                child: Text("Gallary"),
                onTap: (){
                  _openGallary(context);
                },
              ),
              Padding(padding: EdgeInsets.all(8.0)),
              GestureDetector(
                child: Text("Camera"),
                onTap: (){
                  _openCamera(context);
                },
              )
            ],
          ),
        ),
      );
    });
  }
  Future<void> _showconvimg(BuildContext context){
    _conversion(context);
  }
  Widget _decideImageView(){
    if(imageFile == null){
      return Text("no image selected!");

    } else{
      return Image.file(imageFile, width: 400, height: 400,);
    }

  }
  Widget delay_image(){
    if(conv_img == null){
      return Text("no image!");

    } else{
      return Text("put the conversion button");
    }
  }

  Widget conversion_img(){
    if(conv_img != null ){
      conversion = false;
      return Image.file(imageFile, width: 400, height: 400,);

    } else{
      return Text("put the conversion button");
    }
  }

  void _upload() async{
    String base64Image = base64Encode(imageFile.readAsBytesSync());
    var imagebyte = imageFile.readAsBytesSync();
    String fileName = imageFile.path.split("/").last;
//    formdata.add("file", new UploadFileInfo(imageFile, basename(imageFile.path)));

    var formData = FormData();
    formData.files.addAll([
      MapEntry(
        "file",
        MultipartFile.fromBytes(imagebyte,
            filename: "$fileName"),
      ),

    ]);


      var re = await dio.post("http://192.168.0.158:5000/upload_low",data:formData );
//      var uri = Uri.parse("http://192.168.0.158:5000/upload_low");
//      var request = await http.post(uri);



//    var multipartFile = new http.Multipar
//    tFile(‘file’, stream, length,
//    filename: basename(imageFile.path));


//      http.MultipartRequest("POST","192.168.0.158:5000");
//    http.post("192.168.0.158:5000", body: {
//      "file": base64Image,
//      "name": fileName,
//    }).then((res) {
//      print(res.statusCode);
//    }).catchError((err) {
//      print(err);
//    });


  }



  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: <Widget>[
              Padding(padding: EdgeInsets.all(20.0)),
              Container(
                  alignment: Alignment.center,
                  width: 400,
                  height: 400,
//                  child: _decideImageView(),
                  child: _decideImageView(),
              ),

              Padding(padding: EdgeInsets.all(10.0)),


              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  RaisedButton(onPressed: (){
                    _showChoiceDialog(context);
                  },
                    child: Text("Select image"),
                  ),
                  Padding(padding: EdgeInsets.all(5.0)),
                  RaisedButton(onPressed: (){
                    if (imageFile != null) {
                      isLoading = true;

                      _showconvimg(context);
                      Future.delayed(const Duration(seconds: 3), () {
                        isLoading = false;
                        conversion = true;
//                        _showconvimg(context);
                        _upload();
                      });
                    }
                  },
                    child: Text("Conversion"),
                  ),
                ],
              ),

              Padding(padding: EdgeInsets.all(10.0)),
              Container(
                alignment: Alignment.center,
                width: 400,
                height: 400,
                child: conversion ? Center(
                  child: conversion_img(),

                )
                :isLoading ? Center(
                  child: CircularProgressIndicator()
                )
                :delay_image(),

              ),
              Padding(padding: EdgeInsets.all(10.0)),





            ],
          ),
        ),
      ),
    );
  }
}
