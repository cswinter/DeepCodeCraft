FROM hseeberger/scala-sbt:8u222_1.3.5_2.13.1

# Small script for performing shallow git clone of single revision
COPY ./shallow_clone.sh /root/shallow_clone.sh

# Clone and build fixed versions of CodeCraftGame and CodeCraftServer as a straightforward way to download sbt 0.13.16 and populate dependency cache
RUN /root/shallow_clone.sh https://github.com/cswinter/CodeCraftGame.git 92304eb03ef51970a07063ff97e07e1e10f25ff4 /tmp/CodeCraftGame
WORKDIR /tmp/CodeCraftGame
RUN sbt publishLocal
RUN rm -r /tmp/CodeCraftGame
RUN /root/shallow_clone.sh https://github.com/cswinter/CodeCraftServer.git df76892eabb1bb4db0273a60b454a4447716f919 /tmp/CodeCraftServer
WORKDIR /tmp/CodeCraftServer
RUN sbt compile
RUN rm -r /tmp/CodeCraftServer

# Clone and locally publish CodeCraftGame package
RUN /root/shallow_clone.sh https://github.com/cswinter/CodeCraftGame.git 2cc30d6265cd3296849a637189881c6c915fd318 /root/CodeCraftGame
WORKDIR /root/CodeCraftGame
RUN sbt publishLocal

# Clone and build CodeCraftServer
RUN /root/shallow_clone.sh https://github.com/cswinter/CodeCraftServer.git 3b17dec96e181495a19ad2f8216694b500d621b6 /root/CodeCraftServer
WORKDIR /root/CodeCraftServer
RUN sbt compile

CMD sbt run 
