ThisBuild / scalaVersion := "2.13.12"
ThisBuild / version := "0.1.0"
ThisBuild / organization := "lap"

ThisBuild / assemblyMergeStrategy := {
  case PathList("module-info.class")               => MergeStrategy.last
  case path if path.endsWith("/module-info.class") => MergeStrategy.last
  case PathList("META-INF", xs @ _*)               => MergeStrategy.discard
  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}

val chiselVersion = "6.3.0"
val scalaTestVersion = "3.2.16"
val upickleVersion = "3.3.1"

lazy val root = (project in file("."))
  .settings(
    name := "lsq",
    assembly / mainClass := Some("lsq.Main"),
    assembly / assemblyJarName := "lsq-generator.jar",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
      "com.lihaoyi" %% "upickle" % upickleVersion
    ),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations"
    ),
    addCompilerPlugin(
      "org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full
    )
  )
