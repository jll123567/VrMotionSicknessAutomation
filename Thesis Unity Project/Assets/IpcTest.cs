using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Diagnostics;
using System.Net;
using System.Text;
using System.Net.Sockets;

public class IpcTest : MonoBehaviour
{
    public bool working = false;
    string pyScriptPath = "C:/Users/Jill/Desktop/echo.py";
    string response;
    Process p;

    // Start is called before the first frame update
    void Start()
    {
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "python.exe",
            Arguments = pyScriptPath,
            UseShellExecute = false,
            RedirectStandardOutput = false,
            CreateNoWindow = true
        };
        //p = Process.Start(startInfo);;
    }

    // Update is called once per frame
    void Update()
    {

        using Socket sock = new Socket(SocketType.Stream, ProtocolType.Tcp);
        IPAddress host;
        IPAddress.TryParse("127.0.0.1", out host);
        sock.Connect(host, 9696);
        byte[] commandBytes = Encoding.ASCII.GetBytes($"echo:{response}");

        

        int bytesSent = 0;
        while(bytesSent < commandBytes.Length)
        {
            bytesSent += sock.Send(commandBytes, bytesSent, commandBytes.Length - bytesSent, SocketFlags.None);
        }

        byte[] responseBytes = new byte[1024];
        char[] responseChars = new char[1024];
        int bytesReceived;

        while (true)
        {
            bytesReceived = sock.Receive(responseBytes);

            if (bytesReceived == 0) break;
            int charCount = Encoding.ASCII.GetChars(responseBytes, 0, bytesReceived, responseChars, 0);
            if (responseChars[bytesReceived - 1] == ':') break;
        }

        response = new string(responseChars, 0, bytesReceived-1);
        UnityEngine.Debug.Log(response);

    }

    private void OnDestroy()
    {
        p.Close();
    }
}
